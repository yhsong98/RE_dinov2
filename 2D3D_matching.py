import trimesh
import pyrender
import numpy as np
import cv2
import os

mesh = trimesh.load('../data/dino/3D_models/satorius_picus2_5000/picus2_5000.obj', force='mesh', process=False)
mesh.show()
scene = pyrender.Scene()
mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
scene.add(mesh_pyrender)

# Compute mesh center and bounding sphere radius
center = mesh.bounding_box.centroid
radius = mesh.bounding_box.extents.max() * 0.6  # or use mesh.bounding_sphere.radius

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
renderer = pyrender.OffscreenRenderer(640, 480)
# Add a brighter directional light and a point light at the camera position
directions = [
    np.eye(4),  # +Z (already added)
    np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]),  # +X
    np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]),  # -X
    np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]),  # +Y
]
for pose in directions[1:]:
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=6.0)
    scene.add(light, pose=pose)


os.makedirs('./rendered_views', exist_ok=True)

def fibonacci_sphere(samples=120, radius=2.0):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        r = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * r
        z = np.sin(theta) * r
        points.append([radius * x, radius * y, radius * z])
    return np.array(points)

n_views = 120
cam_positions = fibonacci_sphere(n_views, radius=radius * 1.5)  # Zoom in by reducing multiplier

for idx, cam_pos in enumerate(cam_positions):
    cam_pos = cam_pos + center
    target = center
    up = np.array([0, 0, 1])
    z_axis = (cam_pos - target)
    z_axis /= np.linalg.norm(z_axis)
    if np.abs(z_axis[2]) > 0.98:
        up = np.array([0, 1, 0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rot = np.stack([x_axis, -y_axis, z_axis], axis=1)  # Invert y_axis to flip object upright
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = cam_pos

    cam_node = scene.add(camera, pose=pose)
    color, _ = renderer.render(scene)
    scene.remove_node(cam_node)

    cv2.imwrite(f'./rendered_views/view_{idx:04d}.png', cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
    np.save(f'./rendered_views/pose_{idx:04d}.npy', pose)

renderer.delete()