import json
import os
import re
import random
import statistics

from PIL import ImageDraw, ImageFont

PATTERN_FILE = "data/predefined_patterns.json"

def get_font(size=30):
    """Tries to load Arial font, falls back to default font if unavailable."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()

# Sorting letters from right to left, then top to bottom
def sort_letters(predictions):
    return sorted(predictions, key=lambda p: (-p["x"], p["y"]))

# Match the best ruler (longest pattern wins)
def match_ruler_name(sorted_letters, patterns, image_with, image_height):
    """
    Find the best-matching ruler based on longest pattern first.

    - Ensures letters appear in order and are nearby.
    - Uses `are_letters_nearby` to validate sequence.

    Parameters:
    - sorted_letters: List of detected letters sorted in Hebrew order.
    - patterns: Dictionary of ruler patterns.

    Returns:
    - (Matched ruler name, List of matched letter objects)
    """
    best_match = None
    matched_letters = []

    for ruler, pattern_list in sorted(patterns.items(), key=lambda x: -max(len(p.split("-")) for p in x[1])):
        for pattern in sorted(pattern_list, key=lambda p: -len(p.split("-"))):  # Sort patterns by length
            pattern_letters = pattern.split("-")
            temp_matched = []
            idx = 0

            for letter_idx, letter in enumerate(sorted_letters):
                if idx < len(pattern_letters):
                    if pattern_letters[idx] == "*" or letter["class"] == pattern_letters[idx]:
                        if not temp_matched or are_letters_nearby(temp_matched[-1], letter, image_with, image_height):
                            temp_matched.append(letter)
                            idx += 1

            if idx == len(pattern_letters):  # Found a match
                best_match = ruler
                matched_letters = temp_matched
                return best_match, matched_letters  # Return first successful match

    return "Unknown Ruler", []

# Load ruler patterns (multiple patterns per ruler)
def load_patterns():
    """Load predefined and user-defined ruler name patterns."""
    predefined_patterns = {}

    if os.path.exists(PATTERN_FILE):
        try:
            with open(PATTERN_FILE, "r") as file:
                content = file.read().strip()  # Remove any extra spaces/newlines
                user_patterns = json.loads(content) if content else {}  # Handle empty file
        except (json.JSONDecodeError, FileNotFoundError):
            user_patterns = {}  # If file is corrupted, reset to empty dict
    else:
        user_patterns = {}

    # Merge predefined & user-defined patterns
    for ruler, patterns in user_patterns.items():
        if ruler in predefined_patterns:
            predefined_patterns[ruler].extend(patterns)
        else:
            predefined_patterns[ruler] = patterns

    return predefined_patterns

# Save new ruler pattern
def save_pattern(ruler, pattern):
    """Save multiple patterns per ruler and handle empty or corrupted files."""
    os.makedirs("data", exist_ok=True)

    # Ensure the file exists and is not empty
    if os.path.exists(PATTERN_FILE) and os.path.getsize(PATTERN_FILE) > 0:
        try:
            with open(PATTERN_FILE, "r") as file:
                user_patterns = json.load(file)
        except (json.JSONDecodeError, FileNotFoundError):
            user_patterns = {}  # Reset if JSON is corrupted
    else:
        user_patterns = {}  # Initialize empty dictionary

    # Add the new pattern
    if ruler in user_patterns:
        if pattern not in user_patterns[ruler]:  # ✅ Avoid duplicates
            user_patterns[ruler].append(pattern)
    else:
        user_patterns[ruler] = [pattern]

    # Save back to JSON file
    with open(PATTERN_FILE, "w") as file:
        json.dump(user_patterns, file, indent=4)

# Delete a pattern
def delete_pattern(ruler, pattern):
    """Remove a specific pattern from the list."""
    if not os.path.exists(PATTERN_FILE):
        return

    with open(PATTERN_FILE, "r") as file:
        user_patterns = json.load(file)

    if ruler in user_patterns:
        user_patterns[ruler] = [p for p in user_patterns[ruler] if p != pattern]
        if not user_patterns[ruler]:  # Remove empty ruler entry
            del user_patterns[ruler]

    with open(PATTERN_FILE, "w") as file:
        json.dump(user_patterns, file, indent=4)


def visualize_detections(image, predictions, font_size=40, box_color="green", text_color="red"):
    """
    Draws bounding boxes and labels on an image.

    Parameters:
    - image: PIL Image object.
    - predictions: List of detected letters (or matched letters).
    - font_size: User-defined font size.
    - box_color: Color of bounding box.
    - text_color: Color of text.

    Returns:
    - Annotated image with bounding boxes and labels.
    """
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top, right, bottom = x - w / 2, y - h / 2, x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline=box_color, width=4)
        label = f"{pred['class']} ({pred['confidence']:.1%})"

        # Ensure text doesn't go outside the image bounds
        text_x = left
        text_y = max(5, top - font_size - 10)

        # Draw text with user-defined font size
        draw.text((text_x, text_y), label, fill=text_color, font_size=font_size)

    return image


def generate_safe_key(ruler, pattern, index):
    """Generate a unique, safe key for Streamlit elements."""
    safe_pattern = re.sub(r'[^a-zA-Z0-9]', '_', pattern)
    return f"del_{ruler}_{safe_pattern}_{index}"

def are_letters_nearby(box1, box2, image_width, image_height):
    """
    Checks if two letters are near each other based on **image size**.
    - Uses **dynamic row and column thresholds** (similar to Hebrew sorting).
    - Letters in the same row must be close **horizontally**.
    - Letters in different rows must be close **vertically**.

    Parameters:
    - box1, box2: Letter bounding boxes (x, y, width, height).
    - image_width, image_height: The dimensions of the image.

    Returns:
    - **True** if letters are close enough, otherwise **False**.
    """
    x1, y1, w1, h1 = box1["x"], box1["y"], box1["width"], box1["height"]
    x2, y2, w2, h2 = box2["x"], box2["y"], box2["width"], box2["height"]

    # Compute **median letter height & width** (for adaptive thresholding)
    letter_heights = [h1, h2]
    letter_widths = [w1, w2]
    median_height = statistics.median(letter_heights)
    median_width = statistics.median(letter_widths)

    # **Row threshold (Y distance limit)**
    row_threshold = max(median_height * 0.5, image_height * 0.01)

    # **Column threshold (X distance limit)**
    column_threshold = max(median_width * 1.5, image_width * 0.02)

    # Compute distances
    horizontal_distance = abs(x1 - x2)
    vertical_distance = abs(y1 - y2)

    print(f"[DEBUG] Checking proximity: {box1['class']} ↔ {box2['class']}")
    print(f"  Horizontal Distance: {horizontal_distance} (Threshold: {column_threshold})")
    print(f"  Vertical Distance: {vertical_distance} (Threshold: {row_threshold})")

    # Letters must be within **both** horizontal and vertical limits
    return horizontal_distance < column_threshold and vertical_distance < row_threshold

def sort_letters_hebrew(predictions, image_width, image_height):
    """
    Sort letters in Hebrew reading order (right-to-left, then top-to-bottom).
    - Uses **dynamic row thresholding** based on image size.

    Parameters:
    - predictions: List of detected letters with x, y coordinates.
    - image_width, image_height: The size of the image.

    Returns:
    - Sorted list of letters in **Hebrew reading order**.
    """
    print("\n[DEBUG] Original Predictions:")
    for p in predictions:
        print(f"Letter: {p['class']}, X: {p['x']}, Y: {p['y']}")

    # Sort by **y** first (top to bottom)
    predictions.sort(key=lambda p: p["y"])

    print("\n[DEBUG] Sorted by Y (Top to Bottom):")
    for p in predictions:
        print(f"Letter: {p['class']}, X: {p['x']}, Y: {p['y']}")

    # Compute **adaptive row threshold**
    letter_heights = [p["height"] for p in predictions]
    median_letter_height = statistics.median(letter_heights) if letter_heights else 15

    # Calculate **average Y-gap** between consecutive letters
    y_distances = [abs(predictions[i]["y"] - predictions[i-1]["y"]) for i in range(1, len(predictions))]
    avg_y_distance = statistics.mean(y_distances) if y_distances else median_letter_height

    # **Better row thresholding**
    row_threshold = max(median_letter_height * 0.5, avg_y_distance * 0.8, image_height * 0.01)

    print(f"\n[DEBUG] Row threshold set to: {row_threshold}")

    # **Group letters into rows**
    rows = []
    current_row = [predictions[0]]

    for i in range(1, len(predictions)):
        prev_letter = predictions[i - 1]
        current_letter = predictions[i]
        vertical_distance = abs(current_letter["y"] - prev_letter["y"])

        print(f"[DEBUG] Comparing {prev_letter['class']} (Y={prev_letter['y']}) to {current_letter['class']} (Y={current_letter['y']}) | Distance: {vertical_distance}")

        if vertical_distance > row_threshold:
            # Start a **new row**
            rows.append(current_row)
            current_row = [current_letter]
            print(f"[DEBUG] → New row started at letter {current_letter['class']}")
        else:
            # Continue the **same row**
            current_row.append(current_letter)

    # Add the last row
    if current_row:
        rows.append(current_row)

    print("\n[DEBUG] Grouped into Rows:")
    for i, row in enumerate(rows):
        print(f"Row {i + 1}: {[p['class'] for p in row]}")

    # **Sort each row from right to left**
    for row in rows:
        row.sort(key=lambda p: -p["x"])  # Negative x → Right-to-left order

    print("\n[DEBUG] Sorted Rows (Right to Left):")
    for i, row in enumerate(rows):
        print(f"Row {i + 1}: {[p['class'] for p in row]}")

    # **Flatten rows back into a single list**
    sorted_predictions = [letter for row in rows for letter in row]

    print("\n[DEBUG] Final Sorted List:")
    for p in sorted_predictions:
        print(f"Letter: {p['class']}, X: {p['x']}, Y: {p['y']}")

    return sorted_predictions

