import os, json, argparse
import numpy as np
import cv2
import trimesh

def load_pose(pose_json):
    with open(pose_json, "r") as f:
        P = json.load(f)
    K = np.array(P["K"], dtype=np.float32)
    R = np.array(P["R"], dtype=np.float32)
    t = np.array(P["t"], dtype=np.float32).reshape(3,1)
    inliers = P.get("inliers", None)
    return K, R, t, inliers

def unique_edges(faces):
    # edges as sorted tuples to dedupe
    E = set()
    for f in faces:
        i,j,k = int(f[0]), int(f[1]), int(f[2])
        E.add(tuple(sorted((i,j))))
        E.add(tuple(sorted((j,k))))
        E.add(tuple(sorted((k,i))))
    E = np.array(list(E), dtype=np.int32)
    return E

def draw_axes(img, K, R, t, scale=0.1, thickness=2):
    # 3D axes in model frame (X=red, Y=green, Z=blue)
    axes_3d = np.float32([[0,0,0],
                          [scale,0,0],
                          [0,scale,0],
                          [0,0,scale]])
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(axes_3d, rvec, t, K, None)
    pts2d = pts2d.reshape(-1,2).astype(int)
    O, X, Y, Z = pts2d
    cv2.line(img, tuple(O), tuple(X), (0,0,255), thickness)  # X red
    cv2.line(img, tuple(O), tuple(Y), (0,255,0), thickness)  # Y green
    cv2.line(img, tuple(O), tuple(Z), (255,0,0), thickness)  # Z blue
    return img

def main(args):
    # load image (we’ll draw in its original size)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    H, W = img.shape[:2]

    # load pose
    K, R, t, _ = load_pose(args.pose_json)

    # if the image you solved PnP on was resized to 14-multiples,
    # make sure K matches THIS img size. If not, scale K:
    if args.orig_W and args.orig_H:
        Wp, Hp = int(args.orig_W), int(args.orig_H)
        sx, sy = W / float(Wp), H / float(Hp)
        K = K.copy()
        K[0,0] *= sx; K[0,2] *= sx
        K[1,1] *= sy; K[1,2] *= sy

    # load mesh (normalized)
    mesh = trimesh.load(args.mesh, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.faces, dtype=np.int32)
    E = unique_edges(F)

    # project vertices
    rvec, _ = cv2.Rodrigues(R)
    pts2d, _ = cv2.projectPoints(V, rvec, t, K, None)
    pts2d = pts2d.reshape(-1,2)

    overlay = img.copy()

    # draw wireframe (thin lines)
    for (i,j) in E:
        p1 = tuple(np.round(pts2d[i]).astype(int))
        p2 = tuple(np.round(pts2d[j]).astype(int))
        # only draw if both inside the image
        if 0 <= p1[0] < W and 0 <= p1[1] < H and 0 <= p2[0] < W and 0 <= p2[1] < H:
            cv2.line(overlay, p1, p2, (0,255,255), 1)

    # optional: draw sparse vertex points
    if args.draw_points:
        for k in range(0, len(V), max(1, len(V)//2000)):  # cap density
            p = tuple(np.round(pts2d[k]).astype(int))
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(overlay, p, 1, (0,128,255), -1)

    # draw camera axes at model origin
    overlay = draw_axes(overlay, K, R, t, scale=args.axis_size, thickness=2)

    # combine with transparency
    vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    if args.out:
        cv2.imwrite(args.out, vis)
        print(f"[OK] wrote {args.out}")
    else:
        cv2.imshow("pose overlay", vis)
        cv2.waitKey(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pose_json", required=True)
    ap.add_argument("--mesh", required=True, help="model_normalized.obj or .glb")
    ap.add_argument("--out", default=None)
    ap.add_argument("--draw_points", action="store_true")
    ap.add_argument("--axis_size", type=float, default=0.15,
                    help="axis length in normalized model units")
    # If you solved on a resized image (to multiples of 14), but want to draw on the original:
    ap.add_argument("--orig_W", type=int, default=None, help="solver image width (if different)")
    ap.add_argument("--orig_H", type=int, default=None, help="solver image height (if different)")
    args = ap.parse_args()
    main(args)
