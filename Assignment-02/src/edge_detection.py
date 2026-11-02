from pathlib import Path  # Import Path for safe and convenient file paths.

import cv2  # Import OpenCV for image filtering and edge detection.

import numpy as np  # Import NumPy for arrays and mathematical calculations.

import matplotlib.pyplot as plt  # Import Matplotlib for displaying comparison results.


script_dir = Path(__file__).resolve().parent  # Find the directory containing this Python program.

assignment_dir = script_dir.parent  # Move one directory upward to Assignment-02.

input_dir = assignment_dir / "images" / "input"  # Define the input-image folder.

output_dir = assignment_dir / "images" / "output"  # Define the output-image folder.

output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it does not already exist.


input_path = input_dir / "original.png"  # Define the path of the original input image.


image = cv2.imread(str(input_path), cv2.IMREAD_GRAYSCALE)  # Read the original image directly in grayscale.


if image is None:  # Check whether OpenCV successfully loaded the image.

    raise FileNotFoundError(f"Image not found: {input_path}")  # Stop the program if the input image cannot be found.


kernel_x = np.array([[-0.5, 0.0, 0.5]], dtype=np.float64)  # Create the first-order horizontal derivative kernel.


kernel_y = np.array([[-0.5], [0.0], [0.5]], dtype=np.float64)  # Create the first-order vertical derivative kernel.


first_gx = cv2.filter2D(image, cv2.CV_64F, kernel_x, borderType=cv2.BORDER_REPLICATE)  # Calculate first-order horizontal derivative Gx.


first_gy = cv2.filter2D(image, cv2.CV_64F, kernel_y, borderType=cv2.BORDER_REPLICATE)  # Calculate first-order vertical derivative Gy.


first_gx_display = cv2.convertScaleAbs(first_gx)  # Convert Gx absolute values into an 8-bit image for display.


first_gy_display = cv2.convertScaleAbs(first_gy)  # Convert Gy absolute values into an 8-bit image for display.


sobel_gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Calculate the Sobel derivative in the x-direction.


sobel_gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Calculate the Sobel derivative in the y-direction.


sobel_magnitude = np.sqrt((sobel_gx ** 2) + (sobel_gy ** 2))  # Combine Sobel Gx and Gy to calculate non-directional edge strength.


sobel_display = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Normalize Sobel magnitude into the range 0 to 255.


sobel_display = sobel_display.astype(np.uint8)  # Convert the normalized Sobel result into an 8-bit image.


blurred_image = cv2.GaussianBlur(image, (5, 5), 1.0)  # Smooth the image before Laplacian and Canny processing.


laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=1)  # Calculate the second-order Laplacian response.


laplacian_display = cv2.convertScaleAbs(laplacian)  # Convert the absolute Laplacian response into an 8-bit display image.


low_threshold = 100  # Define the lower threshold used by the Canny detector.


high_threshold = 200  # Define the upper threshold used by the Canny detector.


canny_edges = cv2.Canny(blurred_image, low_threshold, high_threshold, L2gradient=True)  # Perform Canny edge detection.


cv2.imwrite(str(output_dir / "01_original_grayscale.png"), image)  # Save the original grayscale image.


cv2.imwrite(str(output_dir / "02_first_order_horizontal_Gx.png"), first_gx_display)  # Save the first-order horizontal derivative result.


cv2.imwrite(str(output_dir / "03_first_order_vertical_Gy.png"), first_gy_display)  # Save the first-order vertical derivative result.


cv2.imwrite(str(output_dir / "04_sobel_magnitude.png"), sobel_display)  # Save the Sobel magnitude edge image.


cv2.imwrite(str(output_dir / "05_laplacian_edges.png"), laplacian_display)  # Save the Laplacian edge-detection result.


cv2.imwrite(str(output_dir / "06_canny_edges.png"), canny_edges)  # Save the Canny edge-detection result.


figure, axes = plt.subplots(2, 3, figsize=(15, 10))  # Create a two-row and three-column comparison figure.


axes[0, 0].imshow(image, cmap="gray")  # Display the original grayscale image.

axes[0, 0].set_title("Original Grayscale Image")  # Add the title for the original image.


axes[0, 1].imshow(first_gx_display, cmap="gray")  # Display the first-order horizontal derivative.

axes[0, 1].set_title("First-Order Gx\nVertical Edge Response")  # Explain that Gx responds strongly to vertical edges.


axes[0, 2].imshow(first_gy_display, cmap="gray")  # Display the first-order vertical derivative.

axes[0, 2].set_title("First-Order Gy\nHorizontal Edge Response")  # Explain that Gy responds strongly to horizontal edges.


axes[1, 0].imshow(sobel_display, cmap="gray")  # Display the non-directional Sobel magnitude.

axes[1, 0].set_title("Sobel Magnitude")  # Add the Sobel result title.


axes[1, 1].imshow(laplacian_display, cmap="gray")  # Display the Laplacian edge result.

axes[1, 1].set_title("Laplacian")  # Add the Laplacian result title.


axes[1, 2].imshow(canny_edges, cmap="gray")  # Display the final Canny edge image.

axes[1, 2].set_title("Canny Edge Detection")  # Add the Canny result title.


for axis in axes.ravel():  # Loop through every image position in the comparison figure.

    axis.axis("off")  # Remove coordinate axes from every displayed image.


plt.tight_layout()  # Automatically improve spacing between the six images.


plt.savefig(output_dir / "07_edge_detection_comparison.png", dpi=300, bbox_inches="tight")  # Save a high-resolution comparison image.


plt.show()  # Display the complete comparison figure on the screen.


plt.close()  # Close the comparison figure after it has been displayed.


print("Image loaded successfully:", input_path)  # Confirm successful loading of the input image.


print("First-order horizontal derivative completed.")  # Confirm Gx processing.


print("First-order vertical derivative completed.")  # Confirm Gy processing.


print("Sobel magnitude edge detection completed.")  # Confirm Sobel processing.


print("Laplacian edge detection completed.")  # Confirm Laplacian processing.


print("Canny edge detection completed.")  # Confirm Canny processing.


print("Canny low threshold:", low_threshold)  # Display the selected lower Canny threshold.


print("Canny high threshold:", high_threshold)  # Display the selected upper Canny threshold.


print("All output images saved in:", output_dir)  # Display the output directory location.