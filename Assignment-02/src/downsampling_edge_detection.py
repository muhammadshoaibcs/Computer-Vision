
from pathlib import Path  # Import Path for managing image and output folder locations.

import cv2  # Import OpenCV for image processing and edge detection.

import numpy as np  # Import NumPy for numerical calculations and image arrays.

import matplotlib.pyplot as plt  # Import Matplotlib for displaying and saving comparison figures.


# --------------------------------------------------
# STEP 1: DEFINE PROJECT DIRECTORIES
# --------------------------------------------------

script_dir = Path(__file__).resolve().parent  # Get the directory containing this Python file.

assignment_dir = script_dir.parent  # Move one level upward to the Assignment-02 directory.

input_dir = assignment_dir / "images" / "input"  # Define the input image directory.

output_dir = assignment_dir / "images" / "output"  # Define the output image directory.

output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it does not exist.


# --------------------------------------------------
# STEP 2: LOAD THE ORIGINAL GRAYSCALE IMAGE
# --------------------------------------------------

input_path = input_dir / "original_downsampling_800x600.png"  # Define the original image filename.

original = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)  # Load the image in grayscale format.

if original is None:  # Check whether the image was loaded successfully.
    raise FileNotFoundError(f"Original image not found: {input_path}")  # Stop execution if the image is missing.

original_height, original_width = original.shape  # Get the original image height and width.

if (original_width, original_height) != (800, 600):  # Verify the required original image dimensions.
    raise ValueError(f"Expected 800x600 image, but received {original_width}x{original_height}.")  # Report incorrect dimensions.

print("Original image loaded successfully.")  # Display a loading confirmation.

print("Original dimensions:", original_width, "x", original_height)  # Display original image dimensions.


# --------------------------------------------------
# STEP 3: SAVE THE ORIGINAL IMAGE
# --------------------------------------------------

cv2.imwrite(str(output_dir / "01_original_800x600.png"), original)  # Save a copy of the original grayscale image.


# --------------------------------------------------
# STEP 4: DEFINE DOWNSAMPLING SETTINGS
# --------------------------------------------------

gaussian_kernel = (5, 5)  # Define a 5x5 Gaussian smoothing kernel.

gaussian_sigma = 1.0  # Define the Gaussian standard deviation.

scale_factor = 0.5  # Reduce image width and height by 50 percent at each level.

canny_low = 100  # Define the lower Canny threshold.

canny_high = 200  # Define the upper Canny threshold.

images = [original]  # Create a list containing the original image.

image_names = ["Original (800x600)"]  # Store the original image label.


# --------------------------------------------------
# STEP 5: PERFORM THREE DOWNSAMPLING OPERATIONS
# --------------------------------------------------

current_image = original.copy()  # Make a copy to preserve the original image.

for level in range(1, 4):  # Repeat the downsampling process three times.

    smoothed_image = cv2.GaussianBlur(  # Apply Gaussian smoothing before reducing resolution.
        current_image,  # Supply the current image.
        gaussian_kernel,  # Supply the 5x5 Gaussian kernel.
        gaussian_sigma  # Supply the Gaussian standard deviation.
    )  # Finish Gaussian smoothing.

    downsampled_image = cv2.resize(  # Resize the smoothed image to half its dimensions.
        smoothed_image,  # Supply the smoothed image.
        None,  # Let OpenCV calculate the output dimensions.
        fx=scale_factor,  # Reduce the width by 50 percent.
        fy=scale_factor,  # Reduce the height by 50 percent.
        interpolation=cv2.INTER_AREA  # Use area-based interpolation for image reduction.
    )  # Finish the downsampling operation.

    current_image = downsampled_image  # Use the new image as the input for the next level.

    images.append(current_image)  # Add the downsampled image to the image list.

    height, width = current_image.shape  # Get the current image dimensions.

    image_names.append(f"Downsample {level} ({width}x{height})")  # Store the image label.

    cv2.imwrite(  # Save the downsampled image.
        str(output_dir / f"0{level + 1}_downsample_{level}.png"),  # Define its output filename.
        current_image  # Supply the image to save.
    )  # Finish saving the image.

    print(f"Downsampling {level}: {width} x {height}")  # Display the current image dimensions.


# --------------------------------------------------
# STEP 6: PERFORM CANNY EDGE DETECTION
# --------------------------------------------------

edge_images = []  # Create a list to store all edge-detection results.

for index, current_image in enumerate(images):  # Process the original and all three downsampled images.

    canny_input = cv2.GaussianBlur(  # Apply light smoothing before Canny detection.
        current_image,  # Supply the current resolution image.
        gaussian_kernel,  # Use the same 5x5 Gaussian kernel.
        gaussian_sigma  # Use the same Gaussian standard deviation.
    )  # Finish the smoothing operation.

    edges = cv2.Canny(  # Perform Canny edge detection.
        canny_input,  # Supply the smoothed grayscale image.
        canny_low,  # Supply the lower threshold.
        canny_high  # Supply the upper threshold.
    )  # Finish Canny edge detection.

    edge_images.append(edges)  # Store the resulting edge image.

    cv2.imwrite(  # Save the edge-detection result.
        str(output_dir / f"edges_{index}_{['original', 'downsample_1', 'downsample_2', 'downsample_3'][index]}.png"),  # Define the output filename.
        edges  # Supply the Canny edge image.
    )  # Finish saving the edge image.

    print(f"Edge detection completed: {image_names[index]}")  # Display the processing confirmation.


# --------------------------------------------------
# STEP 7: DISPLAY ORIGINAL AND DOWNSAMPLED IMAGES
# --------------------------------------------------

figure, axes = plt.subplots(  # Create a figure containing two rows and four columns.
    2,  # Define two rows.
    4,  # Define four columns.
    figsize=(16, 9)  # Set the overall figure size.
)  # Finish figure creation.

for index in range(4):  # Loop through the original image and three downsampled images.

    axes[0, index].imshow(images[index], cmap="gray")  # Display each grayscale image in the first row.

    axes[0, index].set_title(image_names[index])  # Add the corresponding image title.

    axes[0, index].axis("off")  # Hide the coordinate axes.


# --------------------------------------------------
# STEP 8: DISPLAY EDGE-DETECTION RESULTS
# --------------------------------------------------

for index in range(4):  # Loop through all four edge-detection results.

    axes[1, index].imshow(edge_images[index], cmap="gray")  # Display each Canny edge image in the second row.

    axes[1, index].set_title(f"Edge Detection - {image_names[index]}")  # Add the edge image title.

    axes[1, index].axis("off")  # Hide the coordinate axes.


# --------------------------------------------------
# STEP 9: SAVE THE COMPARISON FIGURE
# --------------------------------------------------

plt.tight_layout()  # Automatically adjust spacing between the images.

plt.savefig(  # Save the complete comparison figure.
    output_dir / "09_downsampling_edge_comparison.png",  # Define the comparison image filename.
    dpi=300,  # Set high image resolution for the Word report.
    bbox_inches="tight"  # Remove unnecessary outer whitespace.
)  # Finish saving the comparison figure.

plt.show()  # Display all original, downsampled, and edge images.

plt.close()  # Close the Matplotlib figure.


# --------------------------------------------------
# STEP 10: DISPLAY FINAL EXPERIMENT INFORMATION
# --------------------------------------------------

print("\n========== EXPERIMENT COMPLETED ==========")  # Display the experiment completion heading.

for index, current_image in enumerate(images):  # Loop through all four image resolutions.

    height, width = current_image.shape  # Get the current image dimensions.

    total_pixels = width * height  # Calculate the total number of pixels.

    print(f"{image_names[index]}: {width} x {height}, Pixels: {total_pixels}")  # Display dimensions and pixel count.

print("Canny lower threshold:", canny_low)  # Display the lower Canny threshold.

print("Canny upper threshold:", canny_high)  # Display the upper Canny threshold.

print("All output images saved in:", output_dir)  # Display the final output directory.
