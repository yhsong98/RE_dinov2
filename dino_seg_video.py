import cv2
import torch
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torch.nn.functional as F
import extractor

# Load DINOv2 ViT-b/14 from Torch Hub
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

# Transformations
transform = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

def preprocess(image_path):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def preprocess_mask(mask_path):
    mask = Image.open(mask_path).convert('L')
    mask = mask.resize((14, 14), resample=Image.NEAREST)
    mask_tensor = torch.from_numpy(np.array(mask)).float()
    mask_tensor = (mask_tensor > 128).float()  # Binarize
    return mask_tensor

def extract_features(image_tensor):
    with torch.no_grad():
        feats = model.forward_features(image_tensor.to(device))  # [B, C, H, W]
    return feats


def compute_reference_descriptor(image_path, mask_path):
    image_tensor, _ = preprocess(image_path)
    features_dict = extract_features(image_tensor)

    # Extract and reshape features
    features = features_dict['x_norm_patchtokens']  # [1, N, C]
    B, N, C = features.shape
    H_feat = W_feat = int(N ** 0.5)
    features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]

    # Resize mask to match feature resolution
    mask_tensor = preprocess_mask(mask_path).unsqueeze(0).unsqueeze(0).float().to(device) if mask_path else torch.ones(1, 14, 14).unsqueeze(0).to(device)  # [1, H, W]
                     # [1, 1, H, W]
    mask_tensor = F.interpolate(mask_tensor, size=(H_feat, W_feat), mode='nearest')[0, 0]  # [H_feat, W_feat]

    # Compute masked descriptor
    masked_features = features * mask_tensor.unsqueeze(0)  # [C, H, W]
    descriptor = masked_features.sum(dim=(1, 2)) / (mask_tensor.sum() + 1e-6)
    descriptor = torch.nn.functional.normalize(descriptor.unsqueeze(0), dim=1)  # [1, C]
    return descriptor

def segment_frame(frame_tensor, ref_descriptor):
    with torch.no_grad():
        features = extract_features(frame_tensor)['x_norm_patchtokens']  # [C, H, W]
        B, N, C = features.shape
        H_feat = W_feat = int(N ** 0.5)
        features = features[0].reshape(H_feat, W_feat, C).permute(2, 0, 1)  # [C, H, W]
        C, H, W = features.shape
        flattened = features.view(C, -1).permute(1, 0)  # [H*W, C]
        flattened = torch.nn.functional.normalize(flattened, dim=1)  # normalize
        similarity = torch.matmul(flattened, ref_descriptor.T).view(H, W)  # cosine sim
        similarity = similarity.cpu().numpy()
        print(similarity.shape)
        return similarity

def visualize_segmentation(frame, mask, threshold=0.4):
    mask = (mask > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = frame.copy()
    overlay[mask == 1] = (0, 255, 0)
    blended = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return blended

def run_reference_based_segmentation(ref_img_path, ref_mask_path, camera_id=1):
    ref_descriptor = compute_reference_descriptor(ref_img_path, ref_mask_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)
            frame_tensor = transform(pil_img).unsqueeze(0).to(device)

            similarity_map = segment_frame(frame_tensor, ref_descriptor)
            seg_vis = visualize_segmentation(frame, similarity_map)

            cv2.imshow("Real-time Segmentation", seg_vis)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ==== Run Here ====
if __name__ == "__main__":
    # Provide your paths
    #run_reference_based_segmentation('/home/nml/projects/data/dino/landmark_files/pipette_3D_greybg.png', None)
    run_reference_based_segmentation('/home/nml/projects/data/dino/landmark_files/pipette_s.png', '/home/nml/projects/data/dino/landmark_files/pipette_s_mask.png')