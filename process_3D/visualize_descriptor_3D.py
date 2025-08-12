# make_colored_mesh_from_codebook.py
import os, argparse, numpy as np, trimesh

def pca_to_rgb(X, robust=True):
    X = X.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    # SVD for PCA
    U,S,Vt = np.linalg.svd(X, full_matrices=False)
    pcs = X @ Vt[:3].T  # (N,3)
    if robust:
        lo = np.percentile(pcs, 1, axis=0, keepdims=True)
        hi = np.percentile(pcs, 99, axis=0, keepdims=True)
    else:
        lo = pcs.min(axis=0, keepdims=True)
        hi = pcs.max(axis=0, keepdims=True)
    rng = np.clip(hi - lo, 1e-6, None)
    pcs01 = np.clip((pcs - lo) / rng, 0, 1)
    return (pcs01 * 255).astype(np.uint8)

def main(args):
    desc = np.load(os.path.join(args.dir, "codebook_desc.npy"))  # [Nv, D]
    verts = np.load(os.path.join(args.dir, "codebook_verts.npy")) # [Nv, 3]
    colors = pca_to_rgb(desc, robust=True)
    print(colors)
    rgba = np.concatenate([colors, 255*np.ones((colors.shape[0],1), np.uint8)], axis=1)

    # try to load faces from normalized mesh
    faces = None
    mesh_path = None
    for cand in ["model_normalized.obj", "model_normalized.glb"]:
        p = os.path.join(args.dir, cand)
        if os.path.exists(p):
            mesh_path = p; break

    if mesh_path:
        m = trimesh.load(mesh_path, process=False, force='mesh')
        if isinstance(m, trimesh.Scene):
            m = trimesh.util.concatenate([g for g in m.geometry.values()])
        # assume same topology/order as codebook_verts
        faces = np.asarray(m.faces) if hasattr(m, "faces") and m.faces is not None else None

    if faces is None:
        # vertex-only point cloud PLY
        cloud = trimesh.points.PointCloud(verts, colors=rgba)
        ply_path = os.path.join(args.dir, "colored_mesh_pca.ply")
        cloud.export(ply_path)
        glb_path = os.path.join(args.dir, "colored_mesh_pca.glb")
        cloud.export(glb_path)
        print(f"[OK] wrote point-cloud: {ply_path} and {glb_path}")
    else:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh.visual.vertex_colors = rgba
        ply_path = os.path.join(args.dir, "colored_mesh_pca.ply")
        mesh.export(ply_path)
        glb_path = os.path.join(args.dir, "colored_mesh_pca.glb")
        mesh.export(glb_path)
        print(f"[OK] wrote colored mesh: {ply_path} and {glb_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default='./rendered_views', help="Folder with codebook_desc.npy / codebook_verts.npy (and model_normalized.* if available)")
    main(ap.parse_args())
