import argparse
import os
import torch
from cupy import ndarray
from sklearn.cluster import DBSCAN
from extractor import ViTExtractor
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
import cv2
from scipy.spatial.distance import cdist
from scipy.interpolate import RBFInterpolator
from skimage.transform import PiecewiseAffineTransform, warp
from skimage.measure import label
import random
from PIL import Image
import torch.nn.functional as F
matplotlib.use('Qt5Agg')

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)


def show_similarity_interactive(image_path_a: str, cap, mask_file, image_path_b, num_ref_points: int, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 14, model_type: str = 'dinov2_vits14',
                                num_sim_patches: int = 1, sim_threshold: float = 0.95, num_candidates: int = 10,
                                num_rotations: int = 4, output_csv: bool = False, distance_threshold=10, alpha=0.3, show_landmarks_on_target: bool = False,):
    """
     finding similarity between a descriptor in one image to the all descriptors in the other image.
     :param image_path_a: path to first image.
     :param image_path_b: path to second image.
     :param load_size: size of the smaller edge of loaded images. If None, does not resize.
     :param layer: layer to extract descriptors from.
     :param facet: facet to extract descriptors from.
     :param bin: if True use a log-binning descriptor.
     :param stride: stride of the model.
     :param model_type: type of model to extract descriptors from.
     :param num_sim_patches: number of most similar patches from image_b to plot.
    """
    # extract descriptors
    color_map = [
    "blue", "green",  "cyan", "magenta", "yellow", "black", "orange", "purple", "brown",
    "pink", "gray", "olive", "teal", "navy", "maroon", "lime", "gold", "indigo", "turquoise",
    "violet", "aqua", "coral", "orchid", "salmon", "khaki", "plum", "darkgreen", "darkblue", "crimson"
]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    patch_size = extractor.p#base_model.patch_embed.patch_size[0] if isinstance(extractor.model.patch_embed.patch_size,tuple) else extractor.model.patch_embed.patch_size
    #patch_size=14
    image_batch_a, image_pil_a = extractor.preprocess(image_path_a, load_size)
    descs_a = extractor.extract_descriptors(image_batch_a.to(device), layer, facet, bin, include_cls=True)
    num_patches_a, load_size_a = extractor.num_patches, extractor.load_size

    descs_a = get_masked_descriptors(descs_a.squeeze(1)[:,1:,:], num_patches_a, mask_file, device=device)
    print(descs_a.shape)

    fig, axes = plt.subplots(2, 2, figsize=(30, 30))
    plt.ion()

    axes[0][0].title.set_text('A (reference)')
    axes[0][0].set_axis_off()
    axes[0][1].title.set_text('B (original orientation)')
    axes[0][1].set_axis_off()
    axes[1][0].title.set_text('C (rotated image)')
    axes[1][0].set_axis_off()
    axes[1][1].title.set_text('marked points on original')
    axes[1][1].set_axis_off()

    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    reference_img = axes[0][0].imshow(image_pil_a[0])
    orig_img = axes[0][1].imshow(frame)
    rotated_img = axes[1][0].imshow(frame)
    marked_rotated_img = axes[1][1].imshow(frame)
    visible_patches = []
    fps_counter = 0

    global selected_pts
    fig.canvas.mpl_connect('button_press_event', onclick)

    while True:

        if selected_pts:
            pts = np.asarray(selected_pts)
        else:
            pts = np.asarray([[180, 700]])

        if fps_counter == 0:
            start=time.time()

        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # batch_
        # batch_b_rotations, image_b_rotations = extractor.preprocess(frame, load_size, rotate=num_rotations)
        # descs_b_s = extractor.extract_descriptors(batch_b_rotations.to(device), layer, facet, bin, include_cls=True)
        # #num_patches_b_rotations, load_size_b_rotations = [], []
        # num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        #
        batch_b, image_b = extractor.preprocess(frame, load_size, rotate=False)
        descs_b = extractor.extract_descriptors(batch_b, layer, facet, bin, include_cls=True)
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size
        image_b=image_b[0]
        orig_img.set_data(image_b)
        regions = find_regions_by_reference_object(
            descs_b.squeeze(1)[0,1:,:], descs_a, similarity_threshold=0.93
        )
        try:
            regions = merge_discontinuous_regions_by_feature(descs_b.squeeze(1)[0,1:,:], regions)
        except ValueError:
            pass

        try:
            instance, labels = classify_instance_count(regions, num_patches_b)
        except ValueError:
            labels=[0]*len(regions)

        visualize_instance_labels_cv2(image_b,regions,labels,num_patches_b)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # #mask show, for test usage
        # mask = np.zeros(num_patches_b, dtype=np.uint8)
        # for region in regions:
        #     for idx in region:
        #         y, x = divmod(idx, num_patches_b[1])
        #         mask[y, x] = 1
        #
        # # Resize mask to image size
        # mask_resized = cv2.resize(mask, (image_b.size[1], image_b.size[0]), interpolation=cv2.INTER_NEAREST)
        #
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_b)
        # plt.imshow(mask_resized, alpha=0.5, cmap='jet')  # Overlay mask
        # plt.title("Target Image with Region Mask")
        # plt.axis('off')
        # plt.show()
        #mask show

    plt.close()
    cap.release()

""" taken from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse"""
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

import matplotlib.pyplot as plt
import numpy as np

def visualize_instance_labels(image, regions, labels, num_patches_shape, alpha=0.4):
    """
    Overlay instance regions with different colors on the image.

    :param image: The image as a numpy array (H, W, 3).
    :param regions: List of arrays of token indices per region.
    :param labels: Cluster label for each region.
    :param num_patches_shape: (H, W) patch grid shape.
    :param alpha: Transparency for overlay.
    """
    image = np.array(image)  # Ensure image is a numpy array
    color_map = plt.cm.get_cmap('tab20', np.max(labels)+1)
    mask = np.zeros((num_patches_shape[0], num_patches_shape[1], 3), dtype=np.float32)

    for region, label in zip(regions, labels):
        if label == -1:
            continue  # skip noise
        ys = [idx // num_patches_shape[1] for idx in region]
        xs = [idx % num_patches_shape[1] for idx in region]
        for y, x in zip(ys, xs):
            mask[y, x, :] = color_map(label)[:3]

    # Upscale mask to image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = (image / 255.0) * (1 - alpha) + mask_resized * alpha

    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.axis('off')
    plt.title('Instance Regions Overlay')
    plt.show()

def visualize_instance_labels_cv2(image, regions, labels, num_patches_shape, alpha=0.4):
    """
    Overlay instance regions with different colors on the image using OpenCV.

    :param image: The image as a numpy array (H, W, 3).
    :param regions: List of arrays of token indices per region.
    :param labels: Cluster label for each region.
    :param num_patches_shape: (H, W) patch grid shape.
    :param alpha: Transparency for overlay.
    """
    image = np.array(image)
    mask = np.zeros((num_patches_shape[0], num_patches_shape[1], 3), dtype=np.uint8)
    if len(regions) != 0:
        num_labels = np.max(labels) + 1
        rng = np.random.default_rng(42)
        colors = rng.integers(0, 255, size=(num_labels, 3), dtype=np.uint8)

        for region, label in zip(regions, labels):
            if label == -1:
                continue
            ys = [idx // num_patches_shape[1] for idx in region]
            xs = [idx % num_patches_shape[1] for idx in region]
            for y, x in zip(ys, xs):
                mask[y, x, :] = colors[label]

    # Upscale mask to image size
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(image, 1 - alpha, mask_resized, alpha, 0)
    cv2.imshow('Instance Regions Overlay', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
def merge_discontinuous_regions_by_feature(features_b, regions, feature_eps=0.1, min_samples=1):
    """
    Merge spatially discontinuous regions that belong to the same instance by feature similarity.

    :param features_b: [N, C] tensor of target features.
    :param regions: List of arrays of token indices per region.
    :param feature_eps: DBSCAN epsilon for feature clustering.
    :param min_samples: DBSCAN min_samples.
    :return: List of merged region indices (each as a list).
    """
    # Compute mean feature for each region
    region_features = []
    for region in regions:
        region_feat = features_b[region].mean(dim=0).cpu().numpy()
        region_features.append(region_feat)
    region_features = np.array(region_features)

    # Cluster region features
    clustering = DBSCAN(eps=feature_eps, min_samples=min_samples).fit(region_features)
    labels = clustering.labels_

    # Merge regions by cluster label
    merged_regions = []
    for label in set(labels):
        if label == -1: continue
        merged = np.concatenate([regions[i] for i in range(len(regions)) if labels[i] == label])
        merged_regions.append(merged)
    return merged_regions

def get_masked_descriptors(features_a, num_patches_a, mask_path=None, device='cuda'):
    """
    Extract descriptors from image A at masked locations.

    :param features_a: [1, N_a, C] tensor of reference image features (tokens).
    :param num_patches_a: tuple (H, W) for reference image patch grid.
    :param mask_path: path to mask image for reference.
    :param device: torch device.
    :return: [num_masked, C] tensor of masked descriptors.
    """
    B, N_a, C = features_a.shape
    H_feat, W_feat = num_patches_a
    features_a = features_a[0].reshape(H_feat, W_feat, C)  # [H, W, C]

    if mask_path:
        from PIL import Image
        import numpy as np
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((W_feat, H_feat), resample=Image.NEAREST)
        mask_tensor = torch.from_numpy(np.array(mask)).float()
        mask_tensor = (mask_tensor > 128).float().to(device)
        idx = torch.nonzero(mask_tensor)
        masked_descs_a = features_a[idx[:,0], idx[:,1]]  # [num_masked, C]
    else:
        masked_descs_a = features_a.reshape(-1, C)  # Use all patches if no mask

    return masked_descs_a

def classify_instance_count(regions, num_patches_shape, eps=5, min_samples=1):
    # Compute centroids for each region
    centroids = []
    for region in regions:
        ys, xs = zip(*[(idx // num_patches_shape[1], idx % num_patches_shape[1]) for idx in region])
        centroids.append([np.mean(xs), np.mean(ys)])
    centroids = np.array(centroids)

    # Cluster centroids
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centroids)
    labels = clustering.labels_
    num_instances = len(set(labels)) - (1 if -1 in labels else 0)
    return num_instances, labels  # num_instances: number of detected instances

def detect_object_regions(features, eps=20, min_samples=10):
    # features: [N, C] (tokens/features for one frame)
    # Use DBSCAN clustering on feature space to find object regions
    from sklearn.cluster import DBSCAN
    coords = np.array([[i // features.shape[1], i % features.shape[1]] for i in range(features.shape[0])])
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features.cpu().numpy())
    labels = clustering.labels_
    regions = []
    for label in set(labels):
        if label == -1: continue
        region_indices = np.where(labels == label)[0]
        regions.append(region_indices)
    return regions  # List of arrays of token indices per region

def crop_and_rotate(image, region_indices, num_rotations):
    # Crop region from image and generate rotated crops
    crops = []
    for indices in region_indices:
        # Compute bounding box from indices
        ys, xs = zip(*[(i // image.shape[1], i % image.shape[1]) for i in indices])
        y_min, y_max = min(ys), max(ys)
        x_min, x_max = min(xs), max(xs)
        crop = image[y_min:y_max+1, x_min:x_max+1]
        rotated_crops = [np.rot90(crop, k) for k in range(num_rotations)]
        crops.append(rotated_crops)
    return crops  # List of [rotated_crop_0, ..., rotated_crop_N] per region


def find_regions_by_reference_object(features_b, features_ref, similarity_threshold=0.9, agg='max'):
    """
    Find regions in the target feature map that match the reference object (all its patches) using torch's CosineSimilarity.

    :param features_b: [N, C] tensor of target frame features (tokens).
    :param features_ref: [M, C] tensor of reference object features (tokens).
    :param similarity_threshold: float, threshold for similarity.
    :param agg: 'max' or 'mean' for aggregation.
    :return: List of arrays of token indices per region.
    """
    # Normalize features
    features_b = torch.nn.functional.normalize(features_b, dim=1)
    features_ref = torch.nn.functional.normalize(features_ref, dim=1)

    # Compute pairwise cosine similarity [N, M]
    cos = torch.nn.CosineSimilarity(dim=1)
    sim_matrix = torch.stack([cos(features_b, ref.expand_as(features_b)) for ref in features_ref], dim=1)  # [N, M]

    # Aggregate similarity for each target patch
    if agg == 'max':
        sim_map, _ = torch.max(sim_matrix, dim=1)  # [N]
    else:
        sim_map = torch.mean(sim_matrix, dim=1)    # [N]

    # Threshold to get mask
    mask = (sim_map > similarity_threshold).cpu().numpy().astype(np.uint8)

    # Refine mask
    kernel = np.ones((3, 3), np.uint8)
    mask_refined = cv2.morphologyEx(mask.reshape(-1, 1), cv2.MORPH_CLOSE, kernel)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=1)
    mask_refined = mask_refined.squeeze()

    # Label connected regions
    labeled = label(mask_refined, connectivity=1)
    regions = []
    for region_id in range(1, labeled.max() + 1):
        region_indices = np.where(labeled == region_id)[0]
        if len(region_indices) > 0:
            regions.append(region_indices)
    return regions

def find_regions_by_reference_descriptor(features, ref_descriptor, similarity_threshold=0.9):
    """
    Find regions in the feature map that match the reference descriptor.

    :param features: [N, C] tensor of target frame features (tokens).
    :param ref_descriptor: [1, C] reference descriptor.
    :param similarity_threshold: float, threshold for similarity.
    :return: List of arrays of token indices per region.
    """
    # Normalize features and reference descriptor
    # features = torch.nn.functional.normalize(features, dim=1)
    # ref_descriptor = torch.nn.functional.normalize(ref_descriptor, dim=1)
    cos = torch.nn.CosineSimilarity(dim=1)
    sim_map = cos(features, ref_descriptor.expand_as(features))  # [N]

    # Compute similarity map
    #sim_map = torch.matmul(features, ref_descriptor.T).squeeze(-1)  # [N]

    # Threshold to get mask
    mask = (sim_map > similarity_threshold).cpu().numpy().astype(np.uint8)
    print(mask.shape)
    # Find connected regions (using skimage)
    from skimage.measure import label
    labeled = label(mask, connectivity=1)
    kernel = np.ones((3, 3), np.uint8)  # You can try larger kernels for more smoothing
    mask_refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_refined = cv2.dilate(mask_refined, kernel, iterations=1)

    # Now label connected regions
    labeled = label(mask_refined, connectivity=2)  # 2 for diagonal connectivity

    # Extract regions as before
    regions = []
    for region_id in range(1, labeled.max() + 1):
        region_indices = np.where(labeled == region_id)
        indices_flat = np.ravel_multi_index(region_indices, mask_refined.shape)
        if len(indices_flat) > 0:
            regions.append(indices_flat)
    # regions = []
    # for region_id in range(1, labeled.max() + 1):
    #     region_indices = np.where(labeled == region_id)[0]
    #     if len(region_indices) > 0:
    #         regions.append(region_indices)
    return regions  # List of arrays of token indices per region

def preprocess_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((14, 14), resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(mask)).float()
    mask_tensor = (mask_tensor > 128).float()  # Binarize
    return mask_tensor

def compute_single_token_descriptor(features, num_patches, mask_path=None, device='cuda'):
    B, N, C = features.shape
    H_feat, W_feat = num_patches[0], num_patches[1]
    features = features[0].reshape(H_feat, W_feat, C)  # [H, W, C]

    if mask_path:
        mask_tensor = preprocess_mask(mask_path).to(device)
        mask_tensor = F.interpolate(mask_tensor.unsqueeze(0).unsqueeze(0), size=(H_feat, W_feat), mode='nearest')[0, 0]
        idx = torch.nonzero(mask_tensor)
        # Use the center of the mask as the token location
        y, x = idx.float().mean(dim=0).long()
    else:
        y, x = H_feat // 2, W_feat // 2  # Center token

    descriptor = features[y, x]  # [C]
    descriptor = torch.nn.functional.normalize(descriptor.unsqueeze(0), dim=1)  # [1, C]
    return descriptor

def rotate_landmarks(image_shape, landmarks, angle):
    """
    Rotate landmark coordinates by a given angle around the image center.

    :param image_shape: Tuple (height, width) of the image.
    :param landmarks: List of (x, y) coordinates.
    :param angle: Rotation angle in degrees (counterclockwise).
    :return: List of rotated (x, y) coordinates.
    """
    width, height  = image_shape[:2]
    cx, cy = width / 2, height / 2  # Image center

    # Convert angle to radians
    theta = np.radians(angle)

    # Rotation matrix
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_landmarks = []
    for x, y in landmarks:
        # Translate point to origin
        x_translated, y_translated = x - cx, y - cy

        # Apply rotation
        x_rotated, y_rotated = rotation_matrix @ np.array([x_translated, y_translated])

        # Translate back to image space
        x_new, y_new = x_rotated + cx, y_rotated + cy
        rotated_landmarks.append((int(x_new), int(y_new)))  # Round to nearest pixel

    return rotated_landmarks


def apply_mls(image, src_points, dst_points):
    """
    Apply Moving Least Squares (MLS) deformation to an image.

    :param image: Input image.
    :param src_points: Source landmark points.
    :param dst_points: Target deformed points.
    :return: Warped image.
    """
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    warped = warp(image, tform)
    return (warped * 255).astype(np.uint8)

def onclick(event):
    global selected_pts
    if event.inaxes:
        x, y = event.xdata, event.ydata
        selected_pts = [(int(x), int(y))]
        print(f"New selected point: {selected_pts}")

def classify_landmark(candidate_points, eps=20, min_samples=2):
    """
    Classify a query point based on candidate matches and also return the number of clusters.

    Parameters:
        candidate_points (np.ndarray): An array of shape (N, 2) containing the (x, y)
                                       coordinates of N candidate points on image B.
        eps (float): The maximum distance between two samples for one to be considered
                     as in the neighborhood of the other (adjust based on your image scale).
        min_samples (int): The number of samples in a neighborhood for a point to be
                           considered as a core point.

    Returns:
        result (dict): A dictionary containing:
            - 'label': "landmark" if the candidate points form a single cluster,
                       "non-landmark" if they form multiple clusters,
                       or "uncertain" if no clear cluster is formed.
            - 'num_clusters': The number of clusters detected (ignoring noise points).
    """
    # Run DBSCAN on the candidate points
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
    clustering.fit(candidate_points)
    labels = clustering.labels_  # DBSCAN labels: -1 means noise

    # Count the number of clusters (ignoring noise)
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    num_clusters = len(unique_labels)

    # Decide based on the number of clusters
    if num_clusters == 1:
        label = True  # e.g., nose-like area
    elif num_clusters > 1:
        label = False  # e.g., eyes-like area
    else:
        # If no clusters found, you may decide to return a default label or "uncertain"
        label = False

    return {"label": label, "num_clusters": num_clusters}


def estimate_mls_transform(landmarks_A, landmarks_B):
    """
    Estimate an MLS transformation using Piecewise Affine.

    :param landmarks_A: List of (x, y) landmark coordinates in Image A.
    :param landmarks_B: List of (x, y) corresponding landmark coordinates in Image B.
    :return: Piecewise Affine Transform object.
    """
    landmarks_A = np.array(landmarks_A, dtype=np.float32)
    landmarks_B = np.array(landmarks_B, dtype=np.float32)

    # Create a piecewise affine transformation
    tform = PiecewiseAffineTransform()
    tform.estimate(landmarks_A, landmarks_B)

    return tform




def map_point_mls(point_A, tform):
    """
    Map a point using the MLS transformation.

    :param point_A: (x, y) coordinate in Image A.
    :param tform: MLS transformation object.
    :return: Transformed (x, y) point in Image B.
    """
    point_A = np.array([point_A], dtype=np.float32)  # Reshape input point
    transformed_point = tform(point_A)  # Apply transformation
    return tuple(transformed_point[0])  # Return mapped point


def resolve_ambiguity_mls(point_A, candidates_B, landmarks_A, landmarks_B):
    """
    Resolve ambiguity using MLS warping and landmark correlation.

    :param point_A: Selected (x, y) coordinate in Image A.
    :param candidates_B: List of (x, y) candidate coordinates in Image B.
    :param landmarks_A: List of landmark coordinates in Image A.
    :param landmarks_B: List of corresponding landmark coordinates in Image B.
    :return: Best-matched coordinate in Image B, predicted coordinate, min distance.
    """
    # Compute MLS transformation
    tform = estimate_mls_transform(landmarks_A, landmarks_B)

    # Predict where the point should be in Image B
    predicted_point_B = map_point_mls(point_A, tform)

    if candidates_B:
        # Find the closest candidate to the predicted location
        candidates_B = np.array(candidates_B)
        distances = cdist([predicted_point_B], candidates_B, metric='euclidean')
        min_distance = min(distances[0])
        print("min_distance:", min_distance)

        best_match_idx = np.argmin(distances)
        return tuple(candidates_B[best_match_idx]), predicted_point_B, min_distance
    else:
        return predicted_point_B

def estimate_tps_transform(landmarks_A, landmarks_B):
    """
    Estimate a Thin Plate Spline (TPS) transformation.

    :param landmarks_A: List of (x, y) landmark coordinates in Image A.
    :param landmarks_B: List of (x, y) corresponding landmarks in Image B.
    :return: RBF interpolator for mapping points from A to B.
    """
    landmarks_A = np.array(landmarks_A, dtype=np.float32)
    landmarks_B = np.array(landmarks_B, dtype=np.float32)

    print(landmarks_A.shape)
    print(landmarks_B.shape)

    # Train separate interpolators for X and Y coordinates
    tps_x = RBFInterpolator(landmarks_A, landmarks_B[:, 0], kernel="thin_plate_spline")
    tps_y = RBFInterpolator(landmarks_A, landmarks_B[:, 1], kernel="thin_plate_spline")

    return tps_x, tps_y


def map_point_tps(point_A, tps_x, tps_y):
    """
    Map a point using the TPS transformation.

    :param point_A: (x, y) coordinate in Image A.
    :param tps_x: RBF interpolator for X-coordinates.
    :param tps_y: RBF interpolator for Y-coordinates.
    :return: Transformed (x, y) point in Image B.
    """
    point_A = np.array(point_A).reshape(1, -1)  # Convert to proper shape
    return float(tps_x(point_A)), float(tps_y(point_A))


def resolve_ambiguity_tps(point_A, candidates_B, landmarks_A, landmarks_B):
    """
    Resolve ambiguity using TPS warping and landmark correlation.
    """
    # Compute TPS transformation
    tps_x, tps_y = estimate_tps_transform(landmarks_A, landmarks_B)

    # Predict where the point should be in Image B
    predicted_point_B = map_point_tps(point_A, tps_x, tps_y)

    if candidates_B:
        # # Find the closest candidate to the predicted location
        candidates_B = np.array(candidates_B)
        distances = cdist([predicted_point_B], candidates_B, metric='euclidean')
        min_distance = min(distances[0])
        #print("min_distance:",min_distance)
        best_match_idx = np.argmin(distances)
        # if min(distances[0]) < 10:
        #     return tuple(candidates_B[best_match_idx])  # Best-matched candidate
        # else:
        #     return predicted_point_B
        return tuple(candidates_B[best_match_idx]), predicted_point_B, min_distance
    else:
        return predicted_point_B  # Return predicted location if no candidates are found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Facilitate similarity inspection between two images.')
    parser.add_argument('--image_a', type=str, default="../data/dino/landmark_files/pipette_s.png", help='Path to the reference image.')
    parser.add_argument('--mask_file', default="../data/dino/landmark_files/pipette_s_mask.png", type=str, help="A semantic mask can be added to focus on the target object.")
    parser.add_argument('--image_b', type=str, default="../data/dino/pipette/test_7.png", help='Path to the target images.')
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=14, type=int, help="stride of first convolution layer. small stride -> higher resolution.")
    parser.add_argument('--model_type', default='dinov2_vits14', type=str,
                        help="""type of model to extract. 
                              Choose from [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 | 
                              vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]""")
    parser.add_argument('--facet', default='key', type=str, help="""facet to create descriptors from. 
                                                                       options: ['key' | 'query' | 'value' | 'token']""")
    parser.add_argument('--layer', default=11, type=int, help="layer to create descriptors from.")
    parser.add_argument('--bin', default='False', type=str2bool, help="create a binned descriptor if True.")
    parser.add_argument('--num_sim_patches', default=20, type=int, help="number of closest patches to show.")
    parser.add_argument('--num_ref_points', default=200, type=int, help="number of reference points to show.")
    parser.add_argument('--num_candidates', default=10, type=int, help="number of target point candidates.")
    parser.add_argument('--sim_threshold', default=0.92, type=float, help="similarity threshold.")
    parser.add_argument('--distance_threshold', default=20, type=float, help="distance threshold for TPS.")
    parser.add_argument('--num_rotation', default=4, type=int, help="number of test rotations, 4 or 8 recommended")
    parser.add_argument('--output_csv', default=False, type=str,help="CSV file to save landmark points.")
    args = parser.parse_args()

    with torch.no_grad():
        selected_pts=[]
        landmarks = show_similarity_interactive(args.image_a, cv2.VideoCapture(0),args.mask_file, args.image_b,args.num_ref_points,
                                                args.load_size,
                                                args.layer, args.facet, args.bin,
                                                args.stride, args.model_type, args.num_sim_patches,
                                                args.sim_threshold,  args.num_candidates, args.num_rotation,
                                                args.output_csv,  args.distance_threshold)

