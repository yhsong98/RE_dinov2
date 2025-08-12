import os, json, math, argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F_torch
import cv2

# -------------- DINO extractor --------------
class DinoExtractor(torch.nn.Module):
    def __init__(self, arch="dinov2_vits14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", arch)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def tokens(self, img_bchw: torch.Tensor):
        """
        img_bchw: [B,3,H,W] float in [0,1]
        returns feats [B, N, D] (L2-normed), (Hf, Wf)
        """
        out = self.model.forward_features(img_bchw)
        feats = out["x_norm_patchtokens"]  # [B, N, D], already L2-ish
        feats = F_torch.normalize(feats, dim=-1)
        B, N, D = feats.shape
        Hf = Wf = int(math.sqrt(N))
        assert Hf * Wf == N, f"Non-square token grid? N={N}"
        return feats, (Hf, Wf)

# -------------- token centers to pixel coords --------------
def token_centers(H, W, Hf, Wf, device):
    ys = (torch.arange(Hf, device=device) + 0.5) * (H / Hf)
    xs = (torch.arange(Wf, device=device) + 0.5) * (W / Wf)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    uv = torch.stack([gx, gy], dim=-1).reshape(-1,2)  # (x,y)
    return uv

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load codebook ---
    codebook_dir = args.codebook_dir
    desc = np.load(os.path.join(codebook_dir, "codebook_desc.npy"))  # [Nv, D]
    verts = np.load(os.path.join(codebook_dir, "codebook_verts.npy"))  # [Nv, 3]
    desc_t = torch.from_numpy(desc).to(device).float()
    desc_t = F_torch.normalize(desc_t, dim=1)  # safety
    verts_np = verts.astype(np.float32)

    # --- Camera intrinsics ---
    if args.cameras_json:
        with open(args.cameras_json, "r") as f:
            meta = json.load(f)
        fx = float(meta["intrinsics"]["fx"]); fy = float(meta["intrinsics"]["fy"])
        cx = float(meta["intrinsics"]["cx"]); cy = float(meta["intrinsics"]["cy"])
    else:
        fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy

    K = np.array([[fx, 0,  cx],
                  [0,  fy, cy],
                  [0,   0,  1]], dtype=np.float32)

    # --- Load image ---
    img = Image.open(args.image).convert("RGB")
    W0, H0 = img.width, img.height

    # Resize to nearest multiples of 14
    def nearest_multiple(n, m=14):
        return int(round(n / m) * m)

    W = nearest_multiple(W0, 14)
    H = nearest_multiple(H0, 14)
    if (W, H) != (W0, H0):
        print(f"[INFO] Resizing image from ({W0}, {H0}) → ({W}, {H}) to match patch size")
        img = img.resize((W, H), Image.NEAREST)

    img_t = torch.from_numpy(np.asarray(img)).to(device).float() / 255.0  # [H,W,3]
    img_t = img_t.permute(2,0,1).unsqueeze(0).contiguous()  # [1,3,H,W]

    # --- DINO features ---
    dino = DinoExtractor(args.arch).to(device)
    feats, (Hf, Wf) = dino.tokens(img_t)  # [1, N, D]
    feats = feats[0]  # [N, D]
    N_tokens, D = feats.shape
    uv = token_centers(H, W, Hf, Wf, device)  # [N, 2] (x,y) in pixels

    # --- Cosine similarity to codebook (GPU matmul) ---
    # feats: [N,D], desc_t: [Nv,D]  -> sim: [N,Nv]
    sim = feats @ desc_t.T  # L2-normalized => dot == cosine
    best_sim, best_idx = torch.max(sim, dim=1)  # [N], [N]

    # --- Filter correspondences ---
    keep = best_sim >= args.min_sim
    if args.max_pts > 0 and keep.sum().item() > args.max_pts:
        # take top-k by similarity
        vals, inds = torch.topk(best_sim, k=args.max_pts)
        mask_topk = torch.zeros_like(best_sim, dtype=torch.bool)
        mask_topk[inds] = True
        keep = keep & mask_topk

    if keep.sum().item() < 6:
        print(f"[WARN] Too few matches after filtering: {keep.sum().item()}")
        return

    pts2d = uv[keep].detach().cpu().numpy().astype(np.float32)  # [M,2]
    idx3d = best_idx[keep].detach().cpu().numpy()
    pts3d = verts_np[idx3d]  # [M,3] (normalized-mesh coord frame)

    # --- PnP with RANSAC ---
    dist_coeffs = np.zeros((4,1), dtype=np.float32)  # assume no distortion
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=pts3d,
        imagePoints=pts2d,
        cameraMatrix=K,
        distCoeffs=dist_coeffs,
        iterationsCount=2000,
        reprojectionError=args.ransac_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP
    )

    if not ok or inliers is None or len(inliers) < 6:
        print("[ERR] PnPRansac failed or too few inliers.")
        print(f"Matches: {len(pts3d)}, Inliers: {0 if inliers is None else len(inliers)}")
        return

    # Optional refine with LM on inliers
    if args.refine:
        inl = inliers.reshape(-1)
        cv2.solvePnPRefineLM(
            objectPoints=pts3d[inl],
            imagePoints=pts2d[inl],
            cameraMatrix=K,
            distCoeffs=dist_coeffs,
            rvec=rvec, tvec=tvec
        )

    # Compute reprojection error (px)
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist_coeffs)
    proj = proj.reshape(-1,2)
    err = np.linalg.norm(proj - pts2d, axis=1)
    mean_err = float(err.mean())
    med_err = float(np.median(err))
    inl_count = int(len(inliers))

    # Convert rvec->R
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)

    print("=== Pose (camera in normalized-mesh frame) ===")
    print("R =\n", R)
    print("t =", t)
    print(f"Inliers: {inl_count}/{len(pts3d)}  mean reproj err: {mean_err:.2f}px  median: {med_err:.2f}px")

    # Save pose to json if requested
    if args.out_pose:
        out = {
            "K": K.tolist(),
            "R": R.tolist(),
            "t": t.tolist(),
            "inliers": inl_count,
            "matches": int(len(pts3d)),
            "mean_reproj_err_px": mean_err,
            "median_reproj_err_px": med_err
        }
        with open(args.out_pose, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[OK] Pose saved to {args.out_pose}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default='../data/dino/test/model.png')
    ap.add_argument("--codebook_dir", default='process_3D/rendered_views/', help="Folder containing codebook_desc.npy / codebook_verts.npy")
    ap.add_argument("--arch", default="dinov2_vits14", choices=["dinov2_vits14","dinov2_vitb14"])

    # intrinsics: either from cameras.json or manual
    ap.add_argument("--cameras_json", default='process_3D/rendered_views/cameras.json', help="Optional JSON file with camera intrinsics")
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)

    ap.add_argument("--min_sim", type=float, default=0.5, help="min cosine similarity to keep a 2D-3D match")
    ap.add_argument("--max_pts", type=int, default=1500, help="cap correspondences by top similarity")
    ap.add_argument("--ransac_thresh", type=float, default=2.0, help="RANSAC reprojection error threshold (pixels)")
    ap.add_argument("--refine", action="store_true", help="Levenberg-Marquardt refine on inliers")
    ap.add_argument("--out_pose", default='pose.json', help="Optional JSON file to write the pose")
    args = ap.parse_args()

    # if no cameras_json, require manual intrinsics
    if args.cameras_json is None:
        for k in ["fx","fy","cx","cy"]:
            if getattr(args, k) is None:
                raise SystemExit("Provide --cameras_json or all of --fx --fy --cx --cy")
    main(args)
