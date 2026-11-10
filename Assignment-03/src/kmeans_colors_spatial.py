
"""
Computer Vision - Assignment 03
Experiment: K-Means Clustering

Question 2:
Perform K-means clustering on a color image using:
1. RGB color information
2. RGB color and pixel positions (x, y)

Python 3.9+
"""

# STEP 1: Import required libraries
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.cluster import KMeans


# STEP 2: Define project paths

# Find the Assignment-03 directory automatically.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Define the original input image path.
INPUT_PATH = (
    PROJECT_DIR
    / "images"
    / "input"
    / "kmeans_input_400x300.png"
)

# Define the output directory.
OUTPUT_DIR = PROJECT_DIR / "images" / "output"

# Create the output directory if it does not exist.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# STEP 3: Define experiment parameters

# Number of clusters.
K = 4

# Spatial weights to compare.
SPATIAL_WEIGHTS = [0.2, 0.5, 1.0]

# Fix the random seed for reproducible results.
RANDOM_STATE = 42


# STEP 4: Load the original image

def load_image(image_path):
    """
    Load an image and convert it into RGB format.
    """

    # Check whether the input image exists.
    if not image_path.exists():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    # Open the image and convert it to RGB.
    image = Image.open(image_path).convert("RGB")

    # Convert the image into a NumPy array.
    image_array = np.array(image)

    # Obtain image height, width and channels.
    height, width, channels = image_array.shape

    print("\n========== IMAGE INFORMATION ==========")
    print("Image:", image_path.name)
    print("Width:", width)
    print("Height:", height)
    print("Channels:", channels)
    print("Total pixels:", width * height)

    return image_array


# STEP 5: Extract normalized RGB features

def extract_rgb_features(image_array):
    """
    Convert an RGB image into a two-dimensional
    feature matrix containing R, G and B.
    """

    # Convert pixel values into floating-point numbers.
    rgb = image_array.astype(np.float32)

    # Normalize RGB values from 0-255 to 0-1.
    rgb = rgb / 255.0

    # Reshape (height, width, 3) into (pixels, 3).
    rgb_features = rgb.reshape(-1, 3)

    return rgb_features


# STEP 6: Generate normalized pixel coordinates

def extract_spatial_features(image_array):
    """
    Generate x and y coordinates for every pixel.
    Normalize both coordinates into the range 0-1.
    """

    height, width, _ = image_array.shape

    # Generate x coordinates from 0 to width-1.
    x_coordinates = np.arange(width)

    # Generate y coordinates from 0 to height-1.
    y_coordinates = np.arange(height)

    # Create coordinate grids.
    xx, yy = np.meshgrid(
        x_coordinates,
        y_coordinates
    )

    # Normalize x coordinates.
    xx = xx.astype(np.float32) / max(width - 1, 1)

    # Normalize y coordinates.
    yy = yy.astype(np.float32) / max(height - 1, 1)

    # Flatten the coordinate grids.
    x_flat = xx.reshape(-1, 1)
    y_flat = yy.reshape(-1, 1)

    # Combine x and y into two spatial features.
    spatial_features = np.hstack(
        (x_flat, y_flat)
    )

    return spatial_features


# STEP 7: Perform K-means clustering

def perform_kmeans(features, k):
    """
    Cluster pixels based on the supplied features.
    """

    # Create the K-means model.
    model = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=RANDOM_STATE,
        n_init=10,
        max_iter=300
    )

    # Train the model and assign a cluster to each pixel.
    labels = model.fit_predict(features)

    # Obtain the final cluster centroids.
    centers = model.cluster_centers_

    # Obtain the clustering objective value.
    inertia = model.inertia_

    # Obtain the number of iterations.
    iterations = model.n_iter_

    return labels, centers, inertia, iterations


# STEP 8: Reconstruct the segmented RGB image

def reconstruct_image(
    labels,
    rgb_features,
    height,
    width,
    k
):
    """
    Replace every pixel with the mean RGB color
    of its assigned cluster.

    This works for both RGB-only and RGB+XY
    clustering, while keeping spatial coordinates
    out of the output colors.
    """

    # Prepare a matrix for the reconstructed pixels.
    result = np.zeros_like(rgb_features)

    # Process every cluster separately.
    for cluster_id in range(k):

        # Find pixels belonging to this cluster.
        mask = labels == cluster_id

        # Skip empty clusters.
        if not np.any(mask):
            continue

        # Calculate the average RGB color.
        mean_color = rgb_features[mask].mean(axis=0)

        # Assign the mean color to all cluster pixels.
        result[mask] = mean_color

    # Convert normalized RGB values back to 0-255.
    result = np.clip(result * 255.0, 0, 255)

    # Convert values to unsigned 8-bit integers.
    result = result.astype(np.uint8)

    # Restore the original image dimensions.
    result = result.reshape(height, width, 3)

    return result


# STEP 9: Save an image

def save_image(image_array, filename):
    """
    Save a NumPy RGB array as a PNG image.
    """

    output_path = OUTPUT_DIR / filename

    Image.fromarray(image_array).save(output_path)

    print("Saved:", output_path.name)


# STEP 10: Main experiment

def main():

    print("========== K-MEANS EXPERIMENT ==========")

    # Load the original image.
    image_array = load_image(INPUT_PATH)

    # Extract image dimensions.
    height, width, _ = image_array.shape

    # Extract normalized RGB features.
    rgb_features = extract_rgb_features(image_array)

    # Extract normalized spatial coordinates.
    spatial_features = extract_spatial_features(
        image_array
    )

    print("\n========== FEATURE INFORMATION ==========")
    print("RGB feature shape:", rgb_features.shape)
    print("Spatial feature shape:", spatial_features.shape)

    # Store images for the final comparison figure.
    comparison_images = [image_array]
    comparison_titles = ["Original Image"]

    # EXPERIMENT A: RGB-only K-means

    print("\n========== RGB-ONLY K-MEANS ==========")

    labels, centers, inertia, iterations = perform_kmeans(
        rgb_features,
        K
    )

    rgb_result = reconstruct_image(
        labels,
        rgb_features,
        height,
        width,
        K
    )

    save_image(
        rgb_result,
        "01_rgb_only_kmeans.png"
    )

    print("K:", K)
    print("Iterations:", iterations)
    print("Inertia:", round(inertia, 4))

    comparison_images.append(rgb_result)
    comparison_titles.append("RGB Only")

    # EXPERIMENT B: RGB + Spatial K-means

    for weight in SPATIAL_WEIGHTS:

        print(
            "\n========== SPATIAL WEIGHT:",
            weight,
            "=========="
        )

        # Apply the spatial weight to x and y.
        weighted_spatial = spatial_features * weight

        # Combine RGB and spatial features.
        combined_features = np.hstack(
            (rgb_features, weighted_spatial)
        )

        print(
            "Combined feature shape:",
            combined_features.shape
        )

        # Perform K-means using five features.
        labels, centers, inertia, iterations = perform_kmeans(
            combined_features,
            K
        )

        # Reconstruct the RGB image.
        spatial_result = reconstruct_image(
            labels,
            rgb_features,
            height,
            width,
            K
        )

        # Create a readable filename.
        weight_name = str(weight).replace(".", "_")

        filename = (
            f"02_spatial_weight_{weight_name}.png"
        )

        # Save the segmented image.
        save_image(spatial_result, filename)

        print("K:", K)
        print("Iterations:", iterations)
        print("Inertia:", round(inertia, 4))

        comparison_images.append(spatial_result)
        comparison_titles.append(
            f"RGB + XY (weight={weight})"
        )

    # STEP 11: Create a comparison figure

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9)
    )

    # Flatten the subplot array.
    axes = axes.flatten()

    # Display all experiment images.
    for index, image in enumerate(comparison_images):

        axes[index].imshow(image)

        axes[index].set_title(
            comparison_titles[index]
        )

        axes[index].axis("off")

    # Hide unused subplot spaces.
    for index in range(len(comparison_images), len(axes)):
        axes[index].axis("off")

    plt.tight_layout()

    # Save the final comparison figure.
    comparison_path = (
        OUTPUT_DIR / "03_kmeans_comparison.png"
    )

    plt.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print("Output directory:", OUTPUT_DIR)
    print("Comparison figure:", comparison_path.name)


# STEP 12: Execute the main function

if __name__ == "__main__":
    main()
