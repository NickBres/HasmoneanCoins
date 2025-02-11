import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import json
import os
from utils import (
    sort_letters, match_ruler_name, load_patterns, save_pattern,
    visualize_detections, delete_pattern, generate_safe_key
)

# API Details
ROBOFLOW_MODEL = "hasmonean_coins_letter_detection"
ROBOFLOW_VERSION = "13"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY:
    raise ValueError("Missing API Key! Set the ROBOFLOW_API_KEY environment variable.")

# App Header
st.title("🪙 Hasmonean Coins Recognition")
st.markdown("### Upload a coin image to detect Hebrew inscriptions and identify rulers.")

# Load stored ruler patterns
patterns = load_patterns()

# ---- Upload Section ----
st.header("⬆️ Upload a Coin Image")
uploaded_file = st.file_uploader("📂 Choose an image (JPG/PNG)", type=["jpg", "png"])

if not uploaded_file:
    st.info("Please upload an image to continue.")
    st.stop()

# Read the uploaded image
image_bytes = uploaded_file.getvalue()
image = Image.open(io.BytesIO(image_bytes))

# ---- Confidence Threshold ----
st.header("🔍 Confidence Filter")
st.write("Show only letters the AI is confident about. Lower confidence to see more possible letters.")
confidence_threshold = st.slider("Minimum Confidence", 1, 100, 50)

# ---- Send Image to Roboflow API ----
api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"
response = requests.post(api_url, files={"file": image_bytes})

if response.status_code != 200:
    st.error("❌ Error: Could not process the image. Check your API key and model settings.")
    st.stop()

# Process Response
result = response.json()
predictions = result.get("predictions", [])

# Save JSON result for debugging (only for uploaded images)
os.makedirs("inference_results", exist_ok=True)
json_filename = f"inference_results/{uploaded_file.name.split('.')[0]}.json"
with open(json_filename, "w") as json_file:
    json.dump(result, json_file, indent=4)

# ---- Letter Selection ----
st.header("✅ Select Letters to Display")
st.write("Uncheck incorrect letters to refine the results.")

if predictions:
    # Filter Predictions
    filtered_predictions = [p for p in predictions if p["confidence"] * 100 >= confidence_threshold]
    sorted_predictions = sort_letters(filtered_predictions)

    if not filtered_predictions:
        st.warning("⚠️ No letters detected at the selected confidence level.")
        st.stop()

    # Track current selections
    if "selected_letters" not in st.session_state:
        st.session_state.selected_letters = {f"{p['class']}_{i}" for i, p in enumerate(sorted_predictions)}

    # Toggle all selections
    all_selected = len(st.session_state.selected_letters) == len(sorted_predictions)
    button_label = "Uncheck All" if all_selected else "Check All"

    if st.button(button_label):
        if all_selected:
            st.session_state.selected_letters.clear()
        else:
            st.session_state.selected_letters = {f"{p['class']}_{i}" for i, p in enumerate(sorted_predictions)}

    # Display checkboxes in multiple columns (3 columns)
    cols = st.columns(3)
    letter_visibility = {}

    for i, pred in enumerate(sorted_predictions):
        letter = pred["class"]
        confidence = pred["confidence"] * 100
        key = f"{letter}_{i}"

        with cols[i % 3]:  # Distribute across columns
            letter_visibility[key] = st.checkbox(
                f"{letter} ({confidence:.1f}%)",
                value=(key in st.session_state.selected_letters),
                key=key
            )

    # Update state when checkboxes are toggled
    st.session_state.selected_letters = {
        key for key, visible in letter_visibility.items() if visible
    }

    # Apply filtering
    visible_predictions = [p for i, p in enumerate(sorted_predictions) if f"{p['class']}_{i}" in st.session_state.selected_letters]

    # ---- Display Recognized Letters ----
    st.header("🤖 Recognition Demonstration")
    st.write("Detected letters are shown on the image.")

    font_size = st.slider("Text Size", min_value=5, max_value=30, value=15)
    detected_image = visualize_detections(image.copy(), visible_predictions, font_size=font_size)
    st.image(detected_image, caption="🔍 Detected Letters", use_container_width=True)

    # ---- Collapsible JSON Data ----
    with st.expander("📜 View Raw JSON Data"):
        st.json(result)

    # ---- Ruler Prediction ----
    ruler_name, matched_letters = match_ruler_name(visible_predictions, patterns)

    if ruler_name != "Unknown Ruler":
        st.header("👑 Predicted Ruler")
        st.subheader(f"**{ruler_name}**")
        st.write("The image below highlights only the letters used in the ruler's pattern.")
        ruler_match_image = visualize_detections(image.copy(), matched_letters, font_size=font_size)
        st.image(ruler_match_image, caption=f"Matched Pattern for {ruler_name}", use_container_width=True)

# ---- Pattern Management (Separated Section) ----
st.markdown("---")  # 🔹 Divider for clarity
st.header("⚙️ Pattern Settings")
st.write("Define ruler names based on specific letter patterns.")

# Add new pattern
new_ruler = st.text_input("Enter Ruler Name:")
new_pattern = st.text_input("Enter Pattern (use `*` for any letter, e.g., 'Aleph-*-Daled-Resh')")

if st.button("💾 Save Pattern"):
    if new_ruler and new_pattern:
        save_pattern(new_ruler, new_pattern)
        st.success(f"✅ Saved pattern for {new_ruler}: `{new_pattern}`")
        st.rerun()

# Manage stored patterns
st.markdown("### 📋 Existing Patterns")
for ruler, pattern_list in patterns.items():
    st.write(f"**🏺 {ruler}:**")
    for index, pattern in enumerate(pattern_list):
        col1, col2 = st.columns([4, 1])
        col1.write(f"- `{pattern}`")
        safe_key = generate_safe_key(ruler, pattern, index)
        if col2.button("❌", key=safe_key):
            delete_pattern(ruler, pattern)
            st.rerun()

# Show stored patterns in a collapsible section
with st.expander("📂 View Saved Patterns"):
    st.json(patterns)