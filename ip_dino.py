import argparse
import os
import torch
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
import random
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


def show_similarity_interactive(image_path_a: str, image_folder_path_b: str, mask_file, num_ref_points: int, load_size: int = 224, layer: int = 11,
                                facet: str = 'key', bin: bool = False, stride: int = 14, model_type: str = 'dinov2_vits14',
                                num_sim_patches: int = 1, sim_threshold: float = 0.95, num_candidates: int = 10,
                                num_rotations: int = 4, output_csv: bool = False, distance_threshold=10, alpha=0.3):
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

    if mask_file:
        mask = cv2.resize(cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE),(load_size_a[1],load_size_a[0]))
        coords = cv2.findNonZero(mask)
        if coords is not None:
            coords_list = coords.reshape(-1, 2).tolist()
            landmarks = random.sample(coords_list, num_ref_points)
        else:
            landmarks = []
    else:
        coords = cv2.findNonZero(np.ones((load_size_a[0],load_size_a[1])))
        coords_list = coords.reshape(-1, 2).tolist()
        landmarks = random.sample(coords_list, num_ref_points)

    a_to_a_similarities = chunk_cosine_sim(descs_a, descs_a)
    #a_to_a_curr_similarities = a_to_a_similarities[0, 0, 0, 1:]
    #a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)

    ptses = np.asarray(landmarks)
    landmarks = []
    landmark_idx = []
    for idx, pt in enumerate(ptses):
        y_coor, x_coor = int(pt[1]), int(pt[0])
        new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
        new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
        y_descs_coor = int(new_H / load_size_a[0] * y_coor)
        x_descs_coor = int(new_W / load_size_a[1] * x_coor)

        # get and draw current similarities
        raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
        reveled_desc_idx_including_cls = raveled_desc_idx + 1

        a_to_a_curr_similarities = a_to_a_similarities[0, 0, reveled_desc_idx_including_cls, 1:]
        #a_to_a_curr_similarities = a_to_a_curr_similarities.reshape(num_patches_a)

        center_a_t_candidates = []
        sims, idxes = torch.topk(a_to_a_curr_similarities, num_sim_patches)
        for sim, idx in zip(sims, idxes):
            if sim > sim_threshold:
                a_t_y_descs_coor, a_t_x_descs_coor = torch.div(idx, num_patches_a[1],
                                                           rounding_mode='floor'), idx % \
                                                                                   num_patches_a[1]
                center_a_t = ((a_t_x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                            (a_t_y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                center_a_t_candidates.append([center_a_t[0].cpu().numpy(), center_a_t[1].cpu().numpy()])

        if len(center_a_t_candidates) > 1 and classify_landmark(center_a_t_candidates)['label']:
            landmarks.append(pt)
            landmark_idx.append(reveled_desc_idx_including_cls)


    fig, axes = plt.subplots(2, 2, figsize=(30, 30))

    # axes[1][1].title.set_text('Placeholder')
    # axes[1][1].set_axis_off()
    # axes[1][1].imshow(Image.open('../data/images/placeholder.jpg'))
    if os.path.isdir(image_folder_path_b):
        all_files = os.listdir(image_folder_path_b)
        images = [file for file in all_files if file.endswith(('.jpg', '.png', '.jpeg'))]
        images.sort()
    elif os.path.isfile(image_folder_path_b):
        images = [image_folder_path_b]
    #random.shuffle(images)

    for image_path in images[::-1]:
        start=time.time()
        if os.path.isdir(image_folder_path_b):
            image_path_b = os.path.join(image_folder_path_b, image_path)
        elif os.path.isfile(image_folder_path_b):
            image_path_b = image_folder_path_b

        batch_b_rotations, image_b_rotations = extractor.preprocess(image_path_b, load_size, rotate=num_rotations)

        descs_b_s = extractor.extract_descriptors(batch_b_rotations, layer, facet, bin, include_cls=True)
        #num_patches_b_rotations, load_size_b_rotations = [], []
        num_patches_b, load_size_b = extractor.num_patches, extractor.load_size

        # plot image_a and the chosen patch. if nothing marked chosen patch is cls patch.
        axes[0][0].clear()
        axes[0][0].title.set_text('A (reference)')
        axes[0][0].set_axis_off()
        axes[0][0].imshow(image_pil_a)

        axes[0][1].clear()
        axes[0][1].title.set_text('B (original orientation)')
        axes[0][1].set_axis_off()
        axes[0][1].imshow(batch_b_rotations[0][1])

        axes[1][1].clear()
        axes[1][1].set_axis_off()
        axes[1][1].title.set_text('Marked Points on Original')
        axes[1][1].imshow(batch_b_rotations[0][1])


        #ptses = np.asarray(landmarks)
        similarities_rotations = []

        #test for remote control
        similarities_rotations = chunk_cosine_sim(descs_a, descs_b_s)

        landmark_idx = torch.LongTensor(landmark_idx)

        similarity = similarities_rotations[:,:,landmark_idx,:]
        similarity = similarity.squeeze(1)

        max_batch_value = torch.max(similarity[:,1:,1:], dim=2)[0]
        max_batch_indices = torch.argmax(max_batch_value, dim=0)
        flat_indices = max_batch_indices.view(-1)
        counts = torch.bincount(flat_indices,minlength=similarity.shape[0])
        fittest_index = counts.argmax().item()

        print(fittest_index)
        #
        rotation_degrees = [angle for angle in np.linspace(0, 360, num_rotations, endpoint=False)]
        rotations = {}
        for i,rotation_degree in enumerate(rotation_degrees):
            rotations[i]=rotation_degree
        print('rotation_degree:',rotations[fittest_index])
        print(image_path)


        image_pil_b = image_b_rotations[fittest_index]
        axes[1][0].clear()
        axes[1][0].set_axis_off()
        axes[1][0].title.set_text('B (rotated)')
        axes[1][0].imshow(image_pil_b)

        a_landmark_points=[]
        b_landmark_points=[]
        real_landmark_points=[]
        multi_curr_similarities = similarities_rotations[fittest_index,:,landmark_idx,1:].squeeze(0)

        for landmark_id, curr_similarities in enumerate(multi_curr_similarities):
            center_b_candidates = []
            sims, idxes = torch.topk(curr_similarities.flatten(), num_sim_patches)
            for sim,idx in zip(sims, idxes):
                if sim > sim_threshold:
                    b_y_descs_coor, b_x_descs_coor = torch.div(idx, num_patches_b[1], rounding_mode='floor'), idx % \
                                                     num_patches_b[1]
                    center_b = ((b_x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                                (b_y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    center_b_candidates.append([center_b[0].cpu().numpy(), center_b[1].cpu().numpy()])

            if len(center_b_candidates) > 1 and classify_landmark(center_b_candidates)['label']:
                a_landmark_points.append(landmarks[landmark_id])
                b_landmark_points.append(center_b_candidates[0])


        print('num_landmark_points:',len(real_landmark_points))
        print('num_confident_points:',len(a_landmark_points))

        output_reference = []
        output_rotated_coords = []
        radius_A = image_pil_a.size[0] / 80
        radius_B = image_pil_b.size[0] / 80

        count=0
        for id, pt in enumerate(zip(a_landmark_points,b_landmark_points)):

            patch_a= plt.Circle(pt[0], radius_A, color=color_map[count%len(color_map)],alpha=alpha)
            axes[0][0].add_patch(patch_a)
            output_reference.append(pt[0])
            label = axes[0][0].annotate(str(count), xy=pt[0], fontsize=6, ha="center")

            patch_b = plt.Circle(pt[1], radius_B, color=color_map[count%len(color_map)],alpha=alpha)
            axes[1][0].add_patch(patch_b)
            output_rotated_coords.append(pt[1])
            label = axes[1][0].annotate(str(count), xy=pt[1], fontsize=6, ha="center")
            count += 1

        output_target = []
        landmarks_on_original = rotate_landmarks(image_pil_b.size,output_rotated_coords,rotations[fittest_index])

        for id,pt in enumerate(landmarks_on_original):
            patch_d =plt.Circle(pt,radius_B,color=color_map[id%len(color_map)], alpha=alpha)
            axes[1][1].add_patch(patch_d)
            output_target.append(pt)
            label = axes[1][1].annotate(str(id), xy=pt, fontsize=6, ha="center")


        if output_csv:
            np.savetxt('landmarks_A.csv',output_reference,delimiter=',')
            np.savetxt('landmarks_B.csv',output_target,delimiter=',')
            #np.savetxt('rotated_coords.csv',rotated_coords,delimiter=',')
        plt.draw()

        print('time:', time.time() - start)
        print("-----------")

        #Interactive Part
        pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))
        visible_patches = []
        while len(pts) == 1:
            print('Picked point at:', pts)

            y_coor, x_coor = int(pts[0,1]), int(pts[0,0])
            new_H = patch_size / stride * (load_size_a[0] // patch_size - 1) + 1
            new_W = patch_size / stride * (load_size_a[1] // patch_size - 1) + 1
            y_descs_coor = int(new_H / load_size_a[0] * y_coor)
            x_descs_coor = int(new_W / load_size_a[1] * x_coor)

            # reset previous marks
            for patch in visible_patches:
                patch.remove()
                visible_patches = []

            # draw chosen point
            center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                      (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
            patch = plt.Circle(center, radius_A, color=(1, 0, 0, 0.75))
            axes[0][0].add_patch(patch)
            visible_patches.append(patch)

            raveled_desc_idx = num_patches_a[1] * y_descs_coor + x_descs_coor
            reveled_desc_idx_including_cls = raveled_desc_idx + 1

            curr_similarities = similarities_rotations[fittest_index,0,reveled_desc_idx_including_cls, 1:]
            print(curr_similarities.shape)
            sims, idxs = torch.topk(curr_similarities, num_candidates)
            if sims[0] < sim_threshold:
                b_center=None
                y_descs_coor, x_descs_coor = torch.div(idxs[0], num_patches_b[1], rounding_mode='floor'), idxs[0] % \
                                                                                                      num_patches_b[1]
                center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                          (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                # patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
                #b_cand = [center[0].cpu().numpy(), center[1].cpu().numpy()]

            else:
                b_center = []
                for idx, sim in zip(idxs, sims):
                    y_descs_coor, x_descs_coor = torch.div(idx, num_patches_b[1], rounding_mode='floor'), idx % num_patches_b[1]
                    center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                              (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                    # patch = plt.Circle(center, radius, color=(1, 0, 0, 0.75))
                    b_center.append([center[0].cpu().numpy(), center[1].cpu().numpy()])

            print('Sim:', sims[0])

            try:
                if b_center:
                    best_match_B, predicted_B, min_distance = resolve_ambiguity_mls(pts[0], b_center, output_reference, output_rotated_coords)

                    if min_distance>distance_threshold:
                        patch = plt.Circle(best_match_B, radius_B, color='green')
                        axes[1][0].add_patch(patch)
                        visible_patches.append(patch)
                        color='green'
                    else:
                        patch = plt.Circle(b_center[0], radius_B, color='red')
                        axes[1][0].add_patch(patch)
                        visible_patches.append(patch)
                        color = 'red'
                else:
                    best_match_B = resolve_ambiguity_mls(pts[0], b_center, output_reference, output_rotated_coords)
                    patch = plt.Circle(best_match_B, radius_B, color='blue')
                    axes[1][0].add_patch(patch)
                    visible_patches.append(patch)
                    color = 'blue'

            except:
                y_descs_coor, x_descs_coor = torch.div(idxs[0], num_patches_b[1], rounding_mode='floor'), idxs[0] % \
                                                                                                      num_patches_b[1]
                center = ((x_descs_coor - 1) * stride + stride + patch_size // 2 - .5,
                          (y_descs_coor - 1) * stride + stride + patch_size // 2 - .5)
                patch = plt.Circle((center[0].cpu().numpy(), center[1].cpu().numpy()), radius_B, color='red')
                axes[1][0].add_patch(patch)
                visible_patches.append(patch)
                color = 'red'
            point_on_origin = [patch.center]
            landmarks_on_original = rotate_landmarks(image_pil_b.size, point_on_origin, rotations[fittest_index])
            patch_origin = plt.Circle(landmarks_on_original[0], radius_B, color=color)
            axes[1][1].add_patch(patch_origin)
            visible_patches.append(patch_origin)
            #extent = axes[1][1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            #fig.savefig('ax11_figure.png', bbox_inches=extent)
            plt.draw()
            pts = np.asarray(plt.ginput(1, timeout=-1, mouse_stop=plt.MouseButton.RIGHT, mouse_pop=None))


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
        print("min_distance:",min_distance)
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
    parser.add_argument('--image_b', type=str, default="../data/dino/landmark_files/pipette.png", help='Path to the target images.')
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
    parser.add_argument('--sim_threshold', default=0.95, type=float, help="similarity threshold.")
    parser.add_argument('--distance_threshold', default=10, type=float, help="distance threshold for TPS.")
    parser.add_argument('--num_rotation', default=4, type=int, help="number of test rotations, 4 or 8 recommended")
    parser.add_argument('--output_csv', default=False, type=str,help="CSV file to save landmark points.")
    args = parser.parse_args()

    with torch.no_grad():
        landmarks = show_similarity_interactive(args.image_a, args.image_b, args.mask_file, args.num_ref_points,
                                                args.load_size,
                                                args.layer, args.facet, args.bin,
                                                args.stride, args.model_type, args.num_sim_patches,
                                                args.sim_threshold,  args.num_candidates, args.num_rotation,
                                                args.output_csv,  args.distance_threshold)

