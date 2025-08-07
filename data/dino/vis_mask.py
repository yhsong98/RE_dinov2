import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse


def load_polygon_from_json(json_path):
    """
    Load polygon annotation points from the JSON file.

    :param json_path: Path to the JSON file.
    :return: List of (x, y) points and image size.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract the first annotation result
    annotation = data[0]["annotations"][0]["result"][0]
    width, height = annotation["original_width"], annotation["original_height"]
    polygon_points = np.array(annotation["value"]["points"], dtype=np.float32)

    # Convert percentage-based points to absolute pixel coordinates
    polygon_points[:, 0] = (polygon_points[:, 0] / 100) * width
    polygon_points[:, 1] = (polygon_points[:, 1] / 100) * height

    return polygon_points.astype(int), (width, height)


def create_mask(polygon_points, image_size):
    """
    Create a binary mask from polygon annotation.

    :param polygon_points: List of (x, y) points.
    :param image_size: Tuple (width, height).
    :return: Binary mask.
    """
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)  # Height x Width
    cv2.fillPoly(mask, [polygon_points], color=255)  # Fill the polygon with white
    return mask


def visualize_and_save_mask(json_path, output_path):
    """
    Visualize and save the semantic mask.

    :param json_path: Path to the JSON file.
    :param output_path: Path to save the output image.
    """
    polygon_points, image_size = load_polygon_from_json(json_path)  # Extract polygon data
    mask = create_mask(polygon_points, image_size)  # Create a mask

    # Save the mask
    cv2.imwrite(output_path, mask)
    print(f"Saved mask visualization to {output_path}")

    # Display the mask
    plt.imshow(mask, cmap="gray")
    plt.axis("off")
    plt.title("Semantic Mask")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize and save a semantic mask from a JSON file.")
    parser.add_argument("--json", type=str, default='mouse.json',help="Path to the JSON file storing the mask.")
    parser.add_argument("--output", type=str, default='test/mouse_mask.png', help="Path to save the output image.")

    args = parser.parse_args()

    visualize_and_save_mask(args.json, args.output)
