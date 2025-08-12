import bpy, sys, os, json, math, mathutils, argparse
from math import pi, sqrt

# ---------------- CLI ----------------
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--obj", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--views", type=int, default=300)
ap.add_argument("--imgsz", type=int, default=448)
ap.add_argument("--fov", type=float, default=30.0)
ap.add_argument("--engine", default="AUTO", choices=["AUTO","CYCLES","BLENDER_EEVEE"])
ap.add_argument("--samples", type=int, default=40)
ap.add_argument("--radius", type=float, default=2.0)

# light rig
ap.add_argument("--n_ring", type=int, default=8)
ap.add_argument("--ring_power", type=float, default=100.0)
ap.add_argument("--ring_size", type=float, default=1.5)
ap.add_argument("--fill_power", type=float, default=100.0)
ap.add_argument("--head_power", type=float, default=100.0)
ap.add_argument("--world_strength", type=float, default=0.1)
ap.add_argument("--clean_mesh", type=int, default=1, help="1=fix common issues, 0=report only")
ap.add_argument("--merge_dist", type=float, default=1e-5, help="Merge-by-distance epsilon")

args = ap.parse_args(argv)

os.makedirs(args.out, exist_ok=True)
rgb_dir = os.path.join(args.out, "rgb"); os.makedirs(rgb_dir, exist_ok=True)

# ---------------- scene/init ----------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

engine = args.engine
if engine == "AUTO":
    engine = "CYCLES" if "cycles" in bpy.context.preferences.addons else "BLENDER_EEVEE"

scene.render.image_settings.file_format = 'PNG'
scene.render.resolution_x = args.imgsz
scene.render.resolution_y = args.imgsz
scene.render.resolution_percentage = 100
scene.render.film_transparent = False

if engine == "CYCLES":
    scene.render.engine = 'CYCLES'
    try:
        prefs = bpy.context.preferences.addons['cycles'].preferences
        scene.cycles.device = 'GPU' if any(d.type != 'CPU' and d.use for d in prefs.devices) else 'CPU'
    except Exception:
        scene.cycles.device = 'CPU'
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_denoising = True
else:
    scene.render.engine = 'BLENDER_EEVEE'
    scene.eevee.taa_render_samples = 32
    scene.eevee.use_gtao = True
    scene.eevee.gtao_distance = 0.5
    scene.eevee.gtao_factor = 1.0

# Gentle world light to lift shadows
if scene.world is None:
    scene.world = bpy.data.worlds.new("World")
scene.world.use_nodes = True
bg = scene.world.node_tree.nodes.get("Background")
if bg:
    bg.inputs[1].default_value = float(args.world_strength)

# ---------------- OBJ import ----------------
def robust_import_obj(path):
    if hasattr(bpy.ops.wm, "obj_import"):
        try:
            return bpy.ops.wm.obj_import(filepath=path)
        except Exception as e:
            print("[WARN] wm.obj_import failed:", e)
    if hasattr(bpy.ops.import_scene, "obj"):
        try:
            return bpy.ops.import_scene.obj(filepath=path, axis_forward='-Z', axis_up='Y')
        except Exception as e:
            print("[WARN] import_scene.obj failed:", e)
    raise RuntimeError("No OBJ importer available in this Blender build.")

print(f"[INFO] Importing OBJ: {args.obj}")
robust_import_obj(args.obj)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
if not mesh_objs:
    raise RuntimeError("No mesh objects found after importing OBJ.")
obj = mesh_objs[0]
bpy.context.view_layer.objects.active = obj
obj.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ---------------- normalize to unit cube ----------------
mesh = obj.data
xs = [v.co.x for v in mesh.vertices]
ys = [v.co.y for v in mesh.vertices]
zs = [v.co.z for v in mesh.vertices]
minv = mathutils.Vector((min(xs), min(ys), min(zs)))
maxv = mathutils.Vector((max(xs), max(ys), max(zs)))
center = (minv + maxv) * 0.5
scale = max((maxv - minv).x, (maxv - minv).y, (maxv - minv).z) or 1.0
T = mathutils.Matrix.Translation(-center)
S = mathutils.Matrix.Scale(1.0 / scale, 4)
obj.matrix_world = S @ T
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# ---------------- analyze & fix geometry (open shells / duplicates / normals) ----------------
import bmesh

def analyze_and_fix(obj, do_fix=True, merge_dist=1e-5):
    me = obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    # --- pre stats
    nonman = [e for e in bm.edges if not e.is_manifold]
    loosev = [v for v in bm.verts if len(v.link_edges) == 0]
    # duplicate faces (same vertex set)
    seen = set(); dup_faces = 0
    for f in bm.faces:
        key = tuple(sorted(v.index for v in f.verts))
        if key in seen: dup_faces += 1
        else: seen.add(key)

    print(f"[GEOM] pre: V={len(bm.verts)} F={len(bm.faces)} nonmanifold_edges={len(nonman)} loose_verts={len(loosev)} dup_faces={dup_faces}")

    if do_fix:
        # 1) merge-by-distance (remove doubles)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=float(merge_dist))
        # 2) delete loose verts
        loosev2 = [v for v in bm.verts if len(v.link_edges) == 0]
        if loosev2:
            bmesh.ops.delete(bm, geom=loosev2, context='VERTS')
        # 3) recalc face normals (outside)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)

    # --- post stats
    bm.normal_update()
    nonman2 = [e for e in bm.edges if not e.is_manifold]
    loosev2 = [v for v in bm.verts if len(v.link_edges) == 0]
    print(f"[GEOM] post: V={len(bm.verts)} F={len(bm.faces)} nonmanifold_edges={len(nonman2)} loose_verts={len(loosev2)}")

    bm.to_mesh(me); me.update()
    bm.free()

# run analysis/cleanup
analyze_and_fix(obj, do_fix=bool(args.clean_mesh), merge_dist=args.merge_dist)

# Make materials cull backfaces (prevents “inside seen through” look on thin shells)
for mat in bpy.data.materials:
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = True

# Ensure only the primary mesh renders (duplicates can cause ghosting)
mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
for k, m in enumerate(mesh_objs):
    m.hide_render = (k != 0)
# ---------------- camera + light rig ----------------
cam_data = bpy.data.cameras.new("cam")
cam_data.lens_unit = 'FOV'
cam_data.angle = args.fov * (pi / 180.0)
cam_obj = bpy.data.objects.new("cam_obj", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj

def add_area(name, location, direction, size, energy, color=(1,1,1)):
    light = bpy.data.lights.new(name=name, type='AREA')
    light.shape = 'SQUARE'
    light.size = float(size)
    light.energy = float(energy)
    light.color = color
    objL = bpy.data.objects.new(name, light)
    scene.collection.objects.link(objL)
    objL.location = mathutils.Vector(location)
    dir_vec = mathutils.Vector(direction).normalized()
    rot_quat = dir_vec.to_track_quat('-Z', 'Y')
    objL.rotation_euler = rot_quat.to_euler()
    return objL

# ring lights
ring_R = args.radius * 1.15
for k in range(args.n_ring):
    th = 2.0 * pi * (k / max(1, args.n_ring))
    x = ring_R * math.cos(th); z = ring_R * math.sin(th); y = 0.0
    add_area(f"ring_{k:02d}", (x,y,z), (-x,-y,-z), args.ring_size, args.ring_power)

# top/bottom fill
add_area("fill_top",    (0,  ring_R,  0), (0,-1, 0), args.ring_size*1.2, args.fill_power)
add_area("fill_bottom", (0, -ring_R,  0), (0, 1, 0), args.ring_size*1.2, args.fill_power*0.8)

# head-light attached to camera
head = add_area("head_light", (0, 0, 0.6), (0,0,-1), args.ring_size*0.8, args.head_power)
head.parent = cam_obj


# ---------------- utilities ----------------
def fib_sphere(n, radius):
    pts = []
    golden = pi * (3.0 - sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / max(1, n - 1)) * 2.0
        r = max(0.0, 1.0 - y*y) ** 0.5
        th = golden * i
        x = math.cos(th) * r
        z = math.sin(th) * r
        pts.append((x * radius, y * radius, z * radius))
    return pts

def aim_at(obj_cam, eye, target=(0,0,0)):
    obj_cam.location = mathutils.Vector(eye)
    direction = (mathutils.Vector(target) - obj_cam.location).normalized()
    rot_quat = direction.to_track_quat('-Z', 'Y')
    obj_cam.rotation_euler = rot_quat.to_euler()

# intrinsics
fx = fy = 0.5 * args.imgsz / math.tan(0.5 * args.fov * pi/180.0)
cx = cy = (args.imgsz - 1) * 0.5
meta = {"intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        "image_size": [args.imgsz, args.imgsz],
        "views": []}

# ---------------- render loop ----------------
eyes = fib_sphere(args.views, args.radius)
for i, eye in enumerate(eyes):
    aim_at(cam_obj, eye)
    scene.render.filepath = os.path.join(rgb_dir, f"view_{i:04d}.png")
    bpy.ops.render.render(write_still=True)

    # world->cam using inverse (robust)
    M_cw = cam_obj.matrix_world.inverted()   # 4x4
    Rcw = M_cw.to_3x3()
    tcw = M_cw.to_translation()
    Rt = [[Rcw[0][0], Rcw[0][1], Rcw[0][2], tcw[0]],
          [Rcw[1][0], Rcw[1][1], Rcw[1][2], tcw[1]],
          [Rcw[2][0], Rcw[2][1], Rcw[2][2], tcw[2]]]
    meta["views"].append({"image": f"rgb/view_{i:04d}.png", "Rt": Rt})

# # ---------------- export normalized OBJ & cameras ----------------
# norm_obj = os.path.join(args.out, "model_normalized.obj")
# bpy.ops.export_scene.obj(filepath=norm_obj, use_materials=True, axis_forward=
# --- Export normalized mesh (OBJ if available, else GLB) ---
norm_base = os.path.join(args.out, "model_normalized")

def try_obj_export(path_no_ext):
    if hasattr(bpy.ops.wm, "obj_export"):
        # New 4.x exporter
        return bpy.ops.wm.obj_export(
            filepath=path_no_ext + ".obj",
            export_selected_objects=False,
            export_materials=True,
            forward_axis='NEGATIVE_Z',
            up_axis='Y'
        )
    if hasattr(bpy.ops.export_scene, "obj"):
        # Legacy add-on (not present on your Snap)
        return bpy.ops.export_scene.obj(
            filepath=path_no_ext + ".obj",
            use_materials=True,
            axis_forward='-Z',
            axis_up='Y'
        )
    return None

ok = try_obj_export(norm_base)
if ok is None or ok != {'FINISHED'}:
    # Fallback: glTF/GLB (always available)
    bpy.ops.export_scene.gltf(
        filepath=norm_base + ".glb",
        export_format='GLB',
        export_yup=True
    )
    print(f"[OK] Exported normalized mesh as GLB: {norm_base}.glb")
else:
    print(f"[OK] Exported normalized mesh as OBJ: {norm_base}.obj")
with open(os.path.join(args.out, "cameras.json"), "w") as f:
    json.dump(meta, f, indent=2)

print(f"[OK] Wrote {len(meta['views'])} images to {rgb_dir}")
print(f"[OK] Wrote cameras.json and model_normalized.obj to {args.out}")
