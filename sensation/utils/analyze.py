import numpy as np


def find_dominant_column(image, target_rgb):
    """Finds the column (left, center, or right) with the most occurrences of the specified RGB value.

    Parameters:
    - image: a 3D NumPy array representing the RGB mask (height x width x 3)
    - target_rgb: a list of the RGB values to search for ([R, G, B])

    Returns:
    - 0 if no pixels with the target RGB values are found
    - 1 if the left column contains the most pixels with the target RGB values
    - 2 if the center column contains the most pixels with the target RGB values
    - 3 if the right column contains the most pixels with the target RGB values
    """
    # Get the shape of the image
    height, width, _ = image.shape

    # Convert the target RGB values to a NumPy array
    target_rgb_np = np.array(target_rgb)

    # Create a binary mask where the RGB values match the target values
    mask = np.all(image == target_rgb_np, axis=-1)

    # Define the width of each column
    column_width = width // 3

    # Count the number of matching pixels in each column
    left_count = np.sum(mask[:, :column_width])
    center_count = np.sum(mask[:, column_width:2*column_width])
    right_count = np.sum(mask[:, 2*column_width:])

    # Find the column with the most matching pixels
    max_count = max(left_count, center_count, right_count)

    # Return the corresponding value
    if max_count == 0:
        return 0
    elif max_count == left_count:
        return 1
    elif max_count == center_count:
        return 2
    else:
        return 3
