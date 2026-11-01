from pathlib import Path  # Import Path to create reliable file and folder paths.

import numpy as np  # Import NumPy for gradient and numerical calculations.

import matplotlib.pyplot as plt  # Import Matplotlib for displaying and saving images.

from PIL import Image  # Import Pillow to open and convert the input image.


script_dir = Path(__file__).resolve().parent  # Get the directory where this Python file is located.

assignment_dir = script_dir.parent  # Move one level up to the Assignment-02 directory.

input_dir = assignment_dir / "images" / "input"  # Define the folder containing the input image.

output_dir = assignment_dir / "images" / "output"  # Define the folder where output images will be saved.

output_dir.mkdir(parents=True, exist_ok=True)  # Create the output folder if it does not already exist.


input_path = input_dir / "original.png"  # Define the complete path of the original input image.


image_pil = Image.open(input_path)  # Open the original image using Pillow.

image_gray = image_pil.convert("L")  # Convert the image into grayscale format.

image = np.array(image_gray, dtype=np.float64)  # Convert the grayscale image into a NumPy array.


gy, gx = np.gradient(image)  # Calculate the vertical gradient Gy and horizontal gradient Gx.


magnitude = np.sqrt((gx ** 2) + (gy ** 2))  # Calculate the gradient magnitude for every pixel.


direction_radians = np.arctan2(gy, gx)  # Calculate the gradient direction in radians.

direction_degrees = np.degrees(direction_radians)  # Convert the gradient direction from radians to degrees.


plt.figure(figsize=(10, 8))  # Create a figure for displaying the original grayscale image.

plt.imshow(image, cmap="gray")  # Display the original grayscale image.

plt.title("Original Grayscale Image")  # Add a title above the original image.

plt.axis("off")  # Hide the coordinate axes.

plt.tight_layout()  # Adjust the figure layout automatically.

plt.savefig(output_dir / "01_original_grayscale.png", dpi=300, bbox_inches="tight")  # Save the original grayscale image.

plt.show()  # Display the original grayscale image.

plt.close()  # Close the current figure.


plt.figure(figsize=(10, 8))  # Create a figure for displaying the horizontal gradient.

plt.imshow(gx, cmap="gray")  # Display the horizontal gradient Gx.

plt.title("Horizontal Gradient Gx")  # Add a title above the horizontal gradient image.

plt.axis("off")  # Hide the coordinate axes.

plt.tight_layout()  # Adjust the figure layout automatically.

plt.savefig(output_dir / "02_horizontal_gradient.png", dpi=300, bbox_inches="tight")  # Save the horizontal gradient image.

plt.show()  # Display the horizontal gradient image.

plt.close()  # Close the current figure.


plt.figure(figsize=(10, 8))  # Create a figure for displaying the vertical gradient.

plt.imshow(gy, cmap="gray")  # Display the vertical gradient Gy.

plt.title("Vertical Gradient Gy")  # Add a title above the vertical gradient image.

plt.axis("off")  # Hide the coordinate axes.

plt.tight_layout()  # Adjust the figure layout automatically.

plt.savefig(output_dir / "03_vertical_gradient.png", dpi=300, bbox_inches="tight")  # Save the vertical gradient image.

plt.show()  # Display the vertical gradient image.

plt.close()  # Close the current figure.


plt.figure(figsize=(10, 8))  # Create a figure for displaying the gradient magnitude.

plt.imshow(magnitude, cmap="gray")  # Display the gradient magnitude image.

plt.title("Gradient Magnitude")  # Add a title above the gradient magnitude image.

plt.axis("off")  # Hide the coordinate axes.

plt.tight_layout()  # Adjust the figure layout automatically.

plt.savefig(output_dir / "04_gradient_magnitude.png", dpi=300, bbox_inches="tight")  # Save the gradient magnitude image.

plt.show()  # Display the gradient magnitude image.

plt.close()  # Close the current figure.


step = 15  # Select every fifteenth pixel to prevent too many arrows from overlapping.


y_positions, x_positions = np.mgrid[0:image.shape[0]:step, 0:image.shape[1]:step]  # Generate coordinates where gradient arrows will be drawn.


sample_gx = gx[::step, ::step]  # Select sampled horizontal gradient values for drawing arrows.

sample_gy = gy[::step, ::step]  # Select sampled vertical gradient values for drawing arrows.

sample_magnitude = magnitude[::step, ::step]  # Select sampled gradient magnitude values.


epsilon = 1e-8  # Define a very small number to prevent division by zero.


unit_gx = sample_gx / (sample_magnitude + epsilon)  # Normalize the horizontal gradient component.

unit_gy = sample_gy / (sample_magnitude + epsilon)  # Normalize the vertical gradient component.


plt.figure(figsize=(12, 10))  # Create a figure for displaying gradient-direction arrows.

plt.imshow(image, cmap="gray")  # Display the original grayscale image as the background.


plt.quiver(  # Start drawing the gradient-direction arrows.

    x_positions,  # Provide the horizontal position of each arrow.

    y_positions,  # Provide the vertical position of each arrow.

    unit_gx,  # Provide the normalized horizontal direction of each arrow.

    unit_gy,  # Provide the normalized vertical direction of each arrow.

    angles="xy",  # Interpret arrow directions using image coordinates.

    scale_units="xy",  # Use image coordinate units for arrow scaling.

    scale=0.12  # Control the visible size of the arrows.

)  # Finish drawing the gradient-direction arrows.


plt.title("Gradient Direction with Arrows")  # Add a title above the gradient-direction image.

plt.axis("off")  # Hide the coordinate axes.

plt.tight_layout()  # Adjust the figure layout automatically.

plt.savefig(output_dir / "05_gradient_direction_arrows.png", dpi=300, bbox_inches="tight")  # Save the gradient-direction image.

plt.show()  # Display the gradient-direction arrows.

plt.close()  # Close the current figure.


print("Original image loaded successfully.")  # Confirm that the original image was loaded.

print("Horizontal gradient Gx calculated successfully.")  # Confirm that Gx was calculated.

print("Vertical gradient Gy calculated successfully.")  # Confirm that Gy was calculated.

print("Gradient magnitude calculated successfully.")  # Confirm that the magnitude was calculated.

print("Gradient direction calculated successfully.")  # Confirm that the direction was calculated.

print("Minimum gradient magnitude:", magnitude.min())  # Display the minimum gradient magnitude.

print("Maximum gradient magnitude:", magnitude.max())  # Display the maximum gradient magnitude.

print("Mean gradient magnitude:", magnitude.mean())  # Display the average gradient magnitude.

print("Output images saved in:", output_dir)  # Display the folder where the result images were saved.