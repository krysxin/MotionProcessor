from PIL import Image, ImageDraw
import numpy as np
import random 
import math
import io

from utils import load_json, timer


def find_smallest_square_and_center_points(points):
    points = np.array(points)
    
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    margin_percentage = 0.1
    margin = max(width, height) * margin_percentage
    side_length = max(width, height) + 2 * margin
    
    # Find the center of the bounding box
    center_x = min_x + width / 2
    center_y = min_y + height / 2
    
    # Find the center of the square
    square_center = side_length / 2
    
    # Center the points
    centered_points = points - [center_x, center_y] + [square_center, square_center]
    
    return centered_points, side_length, center_x, center_y

def scale_and_center_points(points, original_square_size, target_size=1024):
    scale_factor = target_size / original_square_size
    scaled_points = points * scale_factor
    # Center the points in the target image size
    offset = (target_size - (original_square_size * scale_factor)) / 2
    scaled_and_centered_points = scaled_points + offset

    # Invert y-axis
    scaled_and_centered_points[:, 1] = target_size - scaled_and_centered_points[:, 1] - offset * 2

    return scaled_and_centered_points, scale_factor

def draw_pattern_on_image(points, image_size=1024, brush_width=15):
    """
    Draw the pattern defined by `points` on a PIL image.
    """
    # Create a blank image with a white background
    image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # # Draw lines between points
    for i in range(len(points) - 1):
        draw.line([tuple(points[i]), tuple(points[i+1])], fill="red", width=brush_width, joint="curve")


    return image


def draw_flower_like_shapes(points, image_size=1024, brush_width=5, scale=1.5):
    """
    Draw the pattern defined by `points` on an image with a transparent background,
    placing flower-like shapes with a more organic appearance at intervals along the line.
    """
    # Create a blank image with a transparent background ('RGBA' mode for transparency)
    image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    # Parameters to control the appearance of the flowers
    min_petals, max_petals = 1, 5
    min_radius, max_radius = round(25*scale), round(35*scale)
    brush_width = round(brush_width * scale)

    min_points_between_shapes = 5
    max_points_between_shapes = 10
    next_shape_at = random.randint(min_points_between_shapes, max_points_between_shapes)

    points_since_last_shape = random.randint(2, len(points))

    for i in range(len(points) - 1):
        draw.line([tuple(points[i]), tuple(points[i+1])], fill="black", width=brush_width)
        points_since_last_shape += 1

        if points_since_last_shape >= next_shape_at:
            points_since_last_shape = 0
            next_shape_at = random.randint(min_points_between_shapes, max_points_between_shapes)

            center_point = points[i+1]
            petals = random.randint(min_petals, max_petals)

            for _ in range(petals):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.randint(min_radius, max_radius)
                offset = random.uniform(0.8, 1.2)  # Offset for center of each petal to create irregularity
                
                # Calculate petal's center point based on angle and offset
                petal_center = (
                    center_point[0] + offset * radius * math.cos(angle),
                    center_point[1] + offset * radius * math.sin(angle)
                )

                # Draw each petal as an ellipse with random size and orientation
                draw.ellipse([
                    petal_center[0] - radius, petal_center[1] - radius,
                    petal_center[0] + radius, petal_center[1] + radius
                ], fill="black", outline="black")

    return image


def add_leaves_around_line(points, image_size=1024, brush_width=5, leaf_count=10):
    """
    Draw the pattern defined by `points` on an image with a transparent background,
    and add elongated leaf-like shapes around a portion of the lower part of the line,
    avoiding the most bottom part.
    """
    image = Image.new("RGBA", (image_size, image_size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)

    for i in range(len(points) - 1):
        draw.line([tuple(points[i]), tuple(points[i+1])], fill="black", width=brush_width)

    # Calculate the range for leaf placement to avoid the most bottom part
    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    lower_part_start = (max_y - min_y) * 0.5 + min_y  # Start of the lower part
    bottom_exclusion_zone = (max_y - lower_part_start) * 0.2  # Exclude bottom 20% of the lower part
    leaf_placement_end = max_y - bottom_exclusion_zone

    for _ in range(leaf_count):
        # Select points only within the specified range for leaf placement
        eligible_points = [pt for pt in points if lower_part_start <= pt[1] < leaf_placement_end]
        if not eligible_points:  # Check if there are no eligible points
            continue  # Skip the iteration if no points are eligible
        leaf_base_point = random.choice(eligible_points)

        leaf_length = random.randint(30, 50)
        leaf_width = random.randint(5, 15)
        angle = random.uniform(-math.pi, math.pi)

        leaf_top_x = leaf_base_point[0] + math.cos(angle) * leaf_length
        leaf_top_y = leaf_base_point[1] + math.sin(angle) * leaf_length

        bbox_left = min(leaf_base_point[0], leaf_top_x) - leaf_width
        bbox_top = min(leaf_base_point[1], leaf_top_y) - leaf_width
        bbox_right = max(leaf_base_point[0], leaf_top_x) + leaf_width
        bbox_bottom = max(leaf_base_point[1], leaf_top_y) + leaf_width

        draw.ellipse([bbox_left, bbox_top, bbox_right, bbox_bottom], fill="green", outline="black")

    return image



def points2lines(data, max_untouch=5):
    """_summary_
    Turn received data from unity to seperate lines
    Return: list of lines, line: list of 2D points
    """
    lines = []  # List to hold all lines
    current_line = []  # Current line being processed
    false_count = 0  # Counter for consecutive falses

    for point in data:
        if point['penTouchingPaper']:
            # If the pen is touching the paper, reset false counter
            false_count = 0
            
            # Append start and end points to the current line
            current_line.append((point['startX'], point['startY']))
            current_line.append((point['endX'], point['endY']))
        else:
            false_count += 1
            # If 5 consecutive points with penTouchingPaper False, start a new line
            if false_count == max_untouch:
                if current_line:  # If the current line is not empty
                    lines.append(current_line)  # Save the finished line
                    current_line = []  # Start a new line
                false_count = 0  # Reset false count after starting a new line

    # Check if there's an unfinished line when data ends
    if current_line:  # If the last line is not empty
        lines.append(current_line)  # Append the last line to the list of lines

    return lines


def draw_base_flower(points, resize=True, byte_format=True):
    centered_points, square_size, center_x, center_y = find_smallest_square_and_center_points(points)
    scaled_points, _ = scale_and_center_points(np.array(centered_points), square_size)
    image = draw_flower_like_shapes(scaled_points)
    if resize:
        square_size = round(square_size)
        image = image.resize((square_size, square_size))

    if not byte_format:
        return image, center_x, center_y
    else:
        with io.BytesIO() as img_byte_arr:
            image.save(img_byte_arr, format='PNG')  # You can choose format as per your requirement (e.g., 'JPEG')
            img_bytes = img_byte_arr.getvalue()

        return img_bytes, center_x, center_y


    

if __name__ == '__main__':
    with timer("Points-to-image"):
        data = load_json("../Data_Unity/3linesRWSID720_2.txt")

        lines = points2lines(data)
        print(len(lines))


        points = np.array(lines[1])
    

        # centered_points, square_size, _, _ = find_smallest_square_and_center_points(points)

        # # Scale and center points within the target 1024x1024 image size
        # scaled_points, _ = scale_and_center_points(np.array(centered_points), square_size)

        # # Draw the pattern on the image
        # image = draw_flower_like_shapes(scaled_points)
        # # image = add_leaves_around_line(scaled_points)

        # # If you need the image as a NumPy array
        # image_array = np.array(image)
        image, _, _ = draw_base_flower(points, byte_format=False)

        # Show the image (if running in an environment that supports displaying images)
        image.show()

