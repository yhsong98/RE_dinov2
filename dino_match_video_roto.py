# import cv2
# import torch
# import torchvision.transforms as T
# import numpy as np
# import functools
# from PIL import Image
# import torch.nn.functional as F
# from sklearn.cluster import KMeans
#
#
# # Load DINOv2 ViT-b/14 from Torch Hub
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
# model.eval()
#
# def get_transform(granularity):
#     return T.Compose([
#         T.Resize((granularity, granularity)),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]),
#     ])
# # Transformations
# def get_cluster_colors(n_clusters, seed=42):
#     rng = np.random.default_rng(seed)
#     return rng.integers(0, 255, size=(n_clusters, 3), dtype=np.uint8)
#
# def preprocess(image_path, granularity=448):
#     image = Image.open(image_path).convert('RGB')
#     transform = get_transform(granularity)
#     return transform(image).unsqueeze(0), image
#
# def preprocess_mask(mask_path, feat_size):
#     mask = Image.open(mask_path).convert('L')
#     mask = mask.resize((feat_size, feat_size), resample=Image.NEAREST)
#     mask_tensor = torch.from_numpy(np.array(mask)).float()
#     mask_tensor = (mask_tensor > 128).float()
#     return mask_tensor
#
# def extract_features(image_tensor):
#     with torch.no_grad():
#         feats = model.forward_features(image_tensor.to(device))  # [B, C, H, W]
#     return feats
#
#
# def compute_reference_descriptor(image_path, mask_path):
#     image_tensor, _ = preprocess(image_path)
#     features_dict = extract_features(image_tensor)
#
#     # Extract and reshape features
#     features = features_dict['x_norm_patchtokens']  # [1, N, C]
#     B, N, C = features.shape
#     H_feat = W_feat = int(N ** 0.5)
#     features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]
#
#     # Resize mask to match feature resolution
#     mask_tensor = preprocess_mask(mask_path).unsqueeze(0).unsqueeze(0).float().to(device) if mask_path else torch.ones(1, 14, 14).unsqueeze(0).to(device)  # [1, H, W]
#                      # [1, 1, H, W]
#     mask_tensor = F.interpolate(mask_tensor, size=(H_feat, W_feat), mode='nearest')[0, 0]  # [H_feat, W_feat]
#
#     # Compute masked descriptor
#     masked_features = features * mask_tensor.unsqueeze(0)  # [C, H, W]
#     descriptor = masked_features.sum(dim=(1, 2)) / (mask_tensor.sum() + 1e-6)
#     descriptor = torch.nn.functional.normalize(descriptor.unsqueeze(0), dim=1)  # [1, C]
#     return descriptor
#
# def segment_frame(frame_tensor, ref_descriptor):
#     with torch.no_grad():
#         features = extract_features(frame_tensor)['x_norm_patchtokens']  # [C, H, W]
#         B, N, C = features.shape
#         H_feat = W_feat = int(N ** 0.5)
#         features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]
#         C, H, W = features.shape
#         flattened = features.view(C, -1).permute(1, 0)  # [H*W, C]
#         flattened = torch.nn.functional.normalize(flattened, dim=1)  # normalize
#         similarity = torch.matmul(flattened, ref_descriptor.T).view(H, W)  # cosine sim
#         similarity = similarity.cpu().numpy()
#         print(similarity.shape)
#         return similarity
#
# def visualize_segmentation(frame, mask, threshold=0.3):
#     mask = (mask > threshold).astype(np.uint8)
#     mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#     overlay = frame.copy()
#     overlay[mask == 1] = (0, 255, 0)
#     blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
#     return blended
#
#
# def compute_reference_descriptors_clustered(image_path, mask_path, n_clusters=3, granularity=448):
#     image_tensor, _ = preprocess(image_path, granularity)
#     features_dict = extract_features(image_tensor)
#     features = features_dict['x_norm_patchtokens']  # [1, N, C]
#     B, N, C = features.shape
#     H_feat = W_feat = int(N ** 0.5)
#     features = features[0].reshape(H_feat, W_feat, C)  # [H, W, C]
#
#     mask_tensor = preprocess_mask(mask_path, H_feat).unsqueeze(0).unsqueeze(0).float().to(device) if mask_path else torch.ones(1, 14, 14).unsqueeze(0).to(device)
#     mask_tensor = F.interpolate(mask_tensor, size=(H_feat, W_feat), mode='nearest')[0, 0]  # [H_feat, W_feat]
#
#     # Get all patch features inside the mask
#     mask_np = mask_tensor.cpu().numpy().astype(bool)
#     patch_feats = features[mask_np]  # [num_masked, C]
#     patch_feats = torch.nn.functional.normalize(patch_feats, dim=1).cpu().numpy()
#
#     # Cluster the features
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(patch_feats)
#     centers = kmeans.cluster_centers_  # [n_clusters, C]
#     centers = torch.tensor(centers, dtype=torch.float32).to(device)
#     centers = torch.nn.functional.normalize(centers, dim=1)
#     return centers  # [n_clusters, C]
#
# def segment_frame_multi_center(frame_tensor, ref_descriptors):
#     with torch.no_grad():
#         features = extract_features(frame_tensor)['x_norm_patchtokens']
#         B, N, C = features.shape
#         H_feat = W_feat = int(N ** 0.5)
#         features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]
#         C, H, W = features.shape
#         flattened = features.view(C, -1).permute(1, 0)  # [H*W, C]
#         flattened = torch.nn.functional.normalize(flattened, dim=1)
#         # Compute similarity to all cluster centers, take max
#         sim = torch.matmul(flattened, ref_descriptors.T)  # [H*W, n_clusters]
#         max_sim, _ = sim.max(dim=1)
#         similarity = max_sim.view(H, W).cpu().numpy()
#         return similarity
#
# def visualize_segmentation_clusters(frame, frame_tensor, ref_descriptors, threshold=0.3, colors=None):
#     # Extract features and compute similarity to all cluster centers
#     with torch.no_grad():
#         features = extract_features(frame_tensor)['x_norm_patchtokens']
#         B, N, C = features.shape
#         H_feat = W_feat = int(N ** 0.5)
#         features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]
#         C, H, W = features.shape
#         flattened = features.view(C, -1).permute(1, 0)  # [H*W, C]
#         flattened = torch.nn.functional.normalize(flattened, dim=1)
#         sim = torch.matmul(flattened, ref_descriptors.T)  # [H*W, n_clusters]
#         max_sim, cluster_idx = sim.max(dim=1)
#         max_sim = max_sim.view(H, W).cpu().numpy()
#         cluster_idx = cluster_idx.view(H, W).cpu().numpy()
#
#     # Assign colors to clusters
#     n_clusters = ref_descriptors.shape[0]
#     if colors is None:
#         colors = get_cluster_colors(n_clusters)
#     mask = (max_sim > threshold)
#     color_mask = np.zeros((H, W, 3), dtype=np.uint8)
#     for k in range(n_clusters):
#         color_mask[(cluster_idx == k) & mask] = colors[k]
#
#     # Upscale mask to frame size
#     color_mask = cv2.resize(color_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
#     overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
#     return overlay
#
# def visualize_reference_clusters(image_path, mask_path, n_clusters=5, granularity=448, colors=None):
#
#     # Preprocess image and extract features
#     image_tensor, pil_image = preprocess(image_path,granularity)
#     features_dict = extract_features(image_tensor)
#     features = features_dict['x_norm_patchtokens']  # [1, N, C]
#     B, N, C = features.shape
#     H_feat = W_feat = int(N ** 0.5)
#     features = features[0].reshape(H_feat, W_feat, C)  # [H, W, C]
#
#     # Prepare mask
#     mask_tensor = preprocess_mask(mask_path, H_feat).unsqueeze(0).unsqueeze(0).float().to(device) if mask_path else torch.ones(1, 14, 14).unsqueeze(0).to(device)
#     mask_tensor = F.interpolate(mask_tensor, size=(H_feat, W_feat), mode='nearest')[0, 0]  # [H_feat, W_feat]
#     mask_np = mask_tensor.cpu().numpy().astype(bool)
#
#     # Cluster features inside mask
#     patch_feats = features[mask_np]  # [num_masked, C]
#     patch_feats = torch.nn.functional.normalize(patch_feats, dim=1).cpu().numpy()
#     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(patch_feats)
#     cluster_labels = np.full((H_feat, W_feat), -1, dtype=np.int32)
#     cluster_labels[mask_np] = kmeans.labels_
#
#     # Assign colors
#     if colors is None:
#         colors = get_cluster_colors(n_clusters, seed=42)
#     color_mask = np.zeros((H_feat, W_feat, 3), dtype=np.uint8)
#     for k in range(n_clusters):
#         color_mask[cluster_labels == k] = colors[k]
#     #print('reference:',colors)
#
#     # Upscale to image size
#     color_mask = cv2.resize(color_mask, pil_image.size, interpolation=cv2.INTER_NEAREST)
#     image_np = np.array(pil_image)
#     overlay = cv2.addWeighted(image_np, 0.6, color_mask, 0.4, 0)
#
#     cv2.imshow("Reference Clusters", overlay)
#
#
# def run_reference_based_segmentation_clustered(ref_img_path, ref_mask_path, camera_id=1, n_clusters=5, granularity=448, colors=None):
#     ref_descriptors = compute_reference_descriptors_clustered(ref_img_path, ref_mask_path, n_clusters, granularity)
#     transform = get_transform(granularity)
#     cap = cv2.VideoCapture(camera_id)
#     if not cap.isOpened():
#         print("Cannot open camera")
#         return
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             pil_img = Image.fromarray(rgb_frame)
#             frame_tensor = transform(pil_img).unsqueeze(0).to(device)
#
#             #similarity_map = segment_frame_multi_center(frame_tensor, ref_descriptors)
#             #seg_vis = visualize_segmentation(frame, similarity_map)
#             seg_vis = visualize_segmentation_clusters(frame, frame_tensor, ref_descriptors, threshold=0.5, colors=colors)
#
#             cv2.imshow("Real-time Segmentation", seg_vis)
#             if cv2.waitKey(1) == 27:  # ESC to quit
#                 break
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#
# # ==== Run Here ====
# if __name__ == "__main__":
#     n_clusters = 10
#     granularity = 896
#     colors = get_cluster_colors(n_clusters, seed=42)
#     #pipette_3D_greybg.png
#     #pipette_3D_greybg_mask.png
#     visualize_reference_clusters('/home/nml/projects/data/dino/landmark_files/pipette_3D_greybg.png'
#                                  ,'/home/nml/projects/data/dino/landmark_files/pipette_3D_greybg_mask.png'
#                                  , n_clusters=n_clusters
#                                  , granularity=granularity
#                                  , colors=colors)
#     run_reference_based_segmentation_clustered(
#         '/home/nml/projects/data/dino/landmark_files/pipette_3D_greybg.png',
#         '/home/nml/projects/data/dino/landmark_files/pipette_3D_greybg_mask.png',
#         camera_id=0,
#         n_clusters=n_clusters,
#         granularity=granularity,
#         colors= colors
#     )
import trimesh
import pyrender
import numpy as np
import cv2
import time

OBJ_PATH = '/home/nml/projects/data/dino/3D_models/satorius_picus2_5000/picus2_5000.obj'
REF_IMAGE_PATH = 'reference_view.png'
MASK_PATH = 'reference_mask.png'

# Load mesh with textures/materials
trimesh_mesh = trimesh.load(OBJ_PATH, force='mesh', process=False)
scene = pyrender.Scene()
mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
directions = [
    np.eye(4),
    np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]),
    np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]),
    np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]),
]
for pose in directions[1:]:
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
    scene.add(light, pose=pose)
scene.add(mesh)

capture_flag = {'capture': False}

class CustomViewer(pyrender.Viewer):
    def on_key_press(self, symbol, modifiers):
        # 121 is the keycode for 'y'
        if symbol == 121:
            capture_flag['capture'] = True
        super().on_key_press(symbol, modifiers)

v = CustomViewer(
    scene,
    use_raymond_lighting=True,
    run_in_thread=False
)

print("Rotate the model. Press 'y' in the viewer window to capture the current view.")

while v.is_active:
    if capture_flag['capture']:
        # Create a new scene and mesh for offscreen rendering
        offscreen_scene = pyrender.Scene()
        offscreen_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
        for pose in directions[1:]:
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=10.0)
            offscreen_scene.add(light, pose=pose)
        offscreen_scene.add(offscreen_mesh)
        # Copy camera pose from viewer
        camera_node = None
        for node in v.scene.get_nodes():
            if isinstance(node.camera, pyrender.Camera):
                camera_node = node
                break

        if camera_node is not None:
            cam = camera_node.camera
            cam_pose = v.scene.get_pose(camera_node)
            offscreen_scene.add(cam, pose=cam_pose)
        renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)
        color, depth = renderer.render(offscreen_scene)
        mask = (depth > 0).astype(np.uint8) * 255

        cv2.imwrite(REF_IMAGE_PATH, cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        cv2.imwrite(MASK_PATH, mask)
        print(f"Saved reference image to {REF_IMAGE_PATH} and mask to {MASK_PATH}")
        capture_flag['capture'] = False
        continue
    time.sleep(0.05)

