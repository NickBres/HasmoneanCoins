import json
import os
import re
import random

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
def match_ruler_name(sorted_letters, patterns):
    """Find the best-matching ruler based on longest pattern first."""
    letter_sequence = [p["class"] for p in sorted_letters]
    best_match = None
    matched_letters = []

    for ruler, pattern_list in sorted(patterns.items(), key=lambda x: -max(len(p.split("-")) for p in x[1])):
        for pattern in sorted(pattern_list, key=lambda p: -len(p.split("-"))):  # Sort patterns by length
            pattern_letters = pattern.split("-")
            temp_matched = []
            idx = 0

            for letter_idx, letter in enumerate(letter_sequence):
                if idx < len(pattern_letters):
                    if pattern_letters[idx] == "*" or letter == pattern_letters[idx]:
                        temp_matched.append(sorted_letters[letter_idx])  # Store full prediction object
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

