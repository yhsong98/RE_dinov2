import os, argparse, math, numpy as np
import trimesh

# Optional headless renderer
def render_turntable_glb(glb_path, out_dir, n_frames=120, w=800, h=600, radius=2.2, elev_deg=15, fov_y_deg=45, make_gif=False):
    import pyrender
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)

    # Load GLB (colored mesh or point cloud)
    scene_tm = trimesh.load(glb_path, force='scene', process=False)
    # Get a Trimesh or list of them
    geometries = []
    if isinstance(scene_tm, trimesh.Scene):
        for geom in scene_tm.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
            elif isinstance(geom, trimesh.points.PointCloud):
                # Convert point cloud to tiny spheres or skip
                # Here: convert to a thin TriMesh using convex hull (fast-ish)
                if len(geom.vertices) >= 4:
                    try:
                        geometries.append(geom.convex_hull)
                    except Exception:
                        pass
    elif isinstance(scene_tm, trimesh.Trimesh):
        geometries = [scene_tm]
    else:
        raise SystemExit("Could not parse GLB into drawable geometry.")

    # Normalize world bbox to get center & scale
    # (This is only for picking a good camera radius; doesn’t modify mesh)
    all_bounds = np.array([g.bounds for g in geometries])
    bb_min = all_bounds[:,0,:].min(axis=0)
    bb_max = all_bounds[:,1,:].max(axis=0)
    center = (bb_min + bb_max) * 0.5
    diag = np.linalg.norm(bb_max - bb_min)
    if diag < 1e-6: diag = 1.0

    # pyrender scene
    scene = pyrender.Scene(bg_color=[0,0,0,0], ambient_light=[0.15,0.15,0.15,1.0])
    for g in geometries:
        # respect vertex colors if present
        mesh = pyrender.Mesh.from_trimesh(g, smooth=False)
        scene.add(mesh)

    # lights
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)
    for ang in [0, 120, 240]:
        node = pyrender.Node(light=light, matrix=np.eye(4))
        # Positioning via parent node rotation later (we’ll rotate scene around Y axis by changing camera pose)
        scene.add_node(node)

    # camera
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_y_deg), aspectRatio=w/float(h))
    cam_node = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(cam_node)

    r = radius if radius > 0 else 2.2
    elev = np.deg2rad(elev_deg)

    # offscreen renderer
    renderer = pyrender.OffscreenRenderer(viewport_width=w, viewport_height=h)

    def look_at(eye, target, up=np.array([0,1,0], np.float32)):
        F = target - eye
        f = F / np.linalg.norm(F)
        u = up / np.linalg.norm(up)
        s = np.cross(f, u); s = s / np.linalg.norm(s)
        u2 = np.cross(s, f)
        M = np.eye(4, dtype=np.float32)
        M[0, :3] = s
        M[1, :3] = u2
        M[2, :3] = -f
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = -eye
        return M @ T

    target = center
    y = center[1] + math.sin(elev) * r
    r_xy = math.cos(elev) * r

    # Render frames
    frames = []
    for i in range(n_frames):
        theta = 2.0 * math.pi * (i / n_frames)
        x = center[0] + r_xy * math.cos(theta)
        z = center[2] + r_xy * math.sin(theta)
        eye = np.array([x, y, z], dtype=np.float32)
        cam_pose = look_at(eye, target)
        scene.set_pose(cam_node, cam_pose)

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        out_png = os.path.join(out_dir, f"turn_{i:04d}.png")
        Image.fromarray(color).save(out_png)
        if make_gif:
            frames.append(Image.fromarray(color))

    if make_gif and frames:
        frames[0].save(os.path.join(out_dir, "turntable.gif"),
                       save_all=True, append_images=frames[1:], duration=40, loop=0)
    renderer.delete()
    print(f"[OK] Saved turntable frames to: {out_dir}")
    if make_gif:
        print(f"[OK] Wrote: {os.path.join(out_dir, 'turntable.gif')}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glb", default='rendered_views/colored_mesh_pca.glb', help="Path to colored_mesh_pca.glb")
    ap.add_argument("--mode", choices=["interactive","turntable"], default="interactive")
    ap.add_argument("--out_dir", default="./turntable_out")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=600)
    ap.add_argument("--radius", type=float, default=2.2)
    ap.add_argument("--elev", type=float, default=15.0, help="elevation in degrees")
    ap.add_argument("--fovy", type=float, default=45.0)
    ap.add_argument("--gif", action="store_true", help="also save a looping GIF")
    args = ap.parse_args()

    if args.mode == "interactive":
        # Open a viewer window (requires a GUI environment)
        scene = trimesh.load(args.glb, force='scene', process=False)
        scene.show()  # Pyglet window; press 'w' for wireframe, drag to orbit
    else:
        # Headless turntable rendering
        try:
            render_turntable_glb(
                glb_path=args.glb,
                out_dir=args.out_dir,
                n_frames=args.frames,
                w=args.width,
                h=args.height,
                radius=args.radius,
                elev_deg=args.elev,
                fov_y_deg=args.fovy,
                make_gif=args.gif
            )
        except ImportError:
            raise SystemExit("pyrender is required for turntable mode. Install with: pip install pyrender PyOpenGL")
    print("[DONE]")

if __name__ == "__main__":
    main()
