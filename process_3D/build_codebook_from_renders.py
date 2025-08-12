import os
import json
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import trimesh

class DinoExtractor(torch.nn.Module):
    def __init__(self, arch="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", arch)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def tokens(self, img_bchw: torch.Tensor):
        out = self.model.forward_features(img_bchw)
        feats = out["x_norm_patchtokens"]  # [B, N, D]
        B, N, D = feats.shape
        Hf = Wf = int(math.sqrt(N))
        feats = F.normalize(feats, dim=-1)
        return feats, (Hf, Wf)

def pixel_rays(fx, fy, cx, cy, H, W, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    z = torch.ones_like(x)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / torch.linalg.norm(dirs, dim=-1, keepdim=True)
    return dirs

def token_centers(H, W, Hf, Wf, device):
    ys = (torch.arange(Hf, device=device) + 0.5) * (H / Hf)
    xs = (torch.arange(Wf, device=device) + 0.5) * (W / Wf)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    uv = torch.stack([gx, gy], dim=-1).reshape(-1, 2)
    return uv

def barycentric_weights(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01 + 1e-12
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load cameras
    with open(os.path.join(args.dir, "cameras.json"), "r") as f:
        meta = json.load(f)
    fx = meta["intrinsics"]["fx"]
    fy = meta["intrinsics"]["fy"]
    cx = meta["intrinsics"]["cx"]
    cy = meta["intrinsics"]["cy"]
    H = W = meta["image_size"][0]

    # Load mesh
    mesh = trimesh.load(os.path.join(args.dir, "model_normalized.obj"), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
    V = mesh.vertices.view(np.ndarray)
    F = mesh.faces.view(np.ndarray)
    Nv = V.shape[0]
    tmesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tmesh.show()
    rmi = trimesh.ray.ray_triangle.RayMeshIntersector(tmesh)

    # Accumulators
    desc_dim = 384 if args.arch == "dinov2_vits14" else 768
    accum = torch.zeros((Nv, desc_dim), dtype=torch.float32, device=device)
    counts = torch.zeros((Nv,), dtype=torch.float32, device=device)

    # DINO
    dino = DinoExtractor(args.arch).to(device)

    # Precompute token centers & pixel ray directions
    uv = token_centers(H, W, args.imgsz_tokens, args.imgsz_tokens, device)
    uv_round = uv.round().long()
    ray_dirs_cam = pixel_rays(fx, fy, cx, cy, H, W, device)

    # Iterate views
    for vinfo in tqdm(meta["views"], desc="Baking"):
        img = Image.open(os.path.join(args.dir, vinfo["image"])).convert("RGB").resize((W, H), Image.BICUBIC)
        img_t = torch.from_numpy(np.asarray(img)).to(device).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)

        feats, (Hf, Wf) = dino.tokens(img_t)
        assert Hf == args.imgsz_tokens and Wf == args.imgsz_tokens, "Token grid mismatch"
        Nt = Hf * Wf
        toks = feats[0]

        # Camera extrinsics
        Rt = np.array(vinfo["Rt"], dtype=np.float32)
        Rt4 = np.eye(4, dtype=np.float32)
        Rt4[:3, :4] = Rt
        Rt4_inv = np.linalg.inv(Rt4)
        Rwc = Rt4_inv[:3, :3]
        twc = Rt4_inv[:3, 3]

        # Ray origins and directions
        dirs_cam = ray_dirs_cam[uv_round[:, 1], uv_round[:, 0], :].cpu().numpy()
        dirs_world = (Rwc @ dirs_cam.T).T
        dirs_world = dirs_world / np.linalg.norm(dirs_world, axis=-1, keepdims=True)
        origins = np.tile(twc[None, :], (Nt, 1))

        hits = rmi.intersects_first(origins, dirs_world)
        valid = hits >= 0
        if valid.sum() == 0:
            continue
        face_idx = hits[valid]
        locs, index_ray, index_tri = rmi.intersects_location(origins[valid], dirs_world[valid])
        row_map = -np.ones(valid.sum(), dtype=np.int64)
        row_map[index_ray] = np.arange(index_ray.size)
        for k in range(locs.shape[0]):
            r = row_map[index_ray[k]]
            if r < 0:
                continue
            f = face_idx[r]
            a, b, c = V[F[f, 0]], V[F[f, 1]], V[F[f, 2]]
            u, v, w = barycentric_weights(locs[k], a, b, c)
            if (u < -1e-4) or (v < -1e-4) or (w < -1e-4):
                continue
            token_id = torch.tensor(np.where(valid)[0][r], device=device)
            ft = toks[token_id]
            for corner, wgt in zip(F[f], [u, v, w]):
                if wgt <= 0:
                    continue
                accum[corner] += wgt * ft
                counts[corner] += wgt

    counts = counts.clamp_min(1e-6).unsqueeze(1)
    codebook = torch.nn.functional.normalize(accum / counts, dim=1)
    np.save(os.path.join(args.dir, "codebook_desc.npy"), codebook.cpu().numpy().astype(np.float32))
    np.save(os.path.join(args.dir, "codebook_verts.npy"), V.astype(np.float32))
    np.save(os.path.join(args.dir, "counts.npy"), counts.cpu().numpy().squeeze(1))
    print("Saved codebook_desc.npy, codebook_verts.npy, counts.npy")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default='rendered_views/', help="Folder with mesh, cameras, and images")
    ap.add_argument("--arch", default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14"])
    ap.add_argument("--imgsz_tokens", type=int, default=32, help="DINO token grid size")
    args = ap.parse_args()
    main(args)