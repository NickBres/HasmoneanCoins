import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import json
import base64
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration  # ✅ Import Roboflow SDK
from utils import (
    sort_letters, match_ruler_name, load_patterns, save_pattern,
    visualize_detections, delete_pattern, generate_safe_key, sort_letters_hebrew
)

# API Details
ROBOFLOW_MODEL = "hasmonean_coins_letter_detection"
ROBOFLOW_VERSION = "14"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY:
    raise ValueError("Missing API Key! Set the ROBOFLOW_API_KEY environment variable.")

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient("https://detect.roboflow.com", ROBOFLOW_API_KEY)

# Set Confidence and Overlay Defaults
CONFIDENCE_THRESHOLD = 0.35  # Default 50%
OVERLAY_THRESHOLD = 0.80  # Default 50%

# Set custom inference configuration
custom_configuration = InferenceConfiguration(
    confidence_threshold=CONFIDENCE_THRESHOLD,
    iou_threshold=OVERLAY_THRESHOLD  # IoU acts as an overlay threshold
)

# Streamlit Web App Config
st.set_page_config(
    page_title="Hasmonean Coins Recognition",
    page_icon="🪙"
)

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
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# ---- Send Image to Roboflow API ----
with CLIENT.use_configuration(custom_configuration):
    result = CLIENT.infer(image_base64, model_id=f"{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}")

# Check API Response
if not result or "predictions" not in result:
    st.error("❌ Error: Could not process the image. Check your API key and model settings.")
    st.stop()

predictions = result["predictions"]

# ---- Letter Selection ----
st.header("✅ Select Letters to Display")
st.write("Uncheck incorrect letters to refine the results.")

if predictions:
    # Filter Predictions Based on Confidence
    sorted_predictions = sort_letters_hebrew(predictions, image.width, image.height)

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
    ruler_name, matched_letters, matched_pattern = match_ruler_name(visible_predictions, patterns, image.width,
                                                                    image.height)

    if ruler_name != "Unknown Ruler":
        st.header("👑 Predicted Ruler")
        st.subheader(f"**{ruler_name}**")

        # Display the matched pattern **directly from match_ruler_name()**
        if matched_pattern:
            st.write(f"**Matched Pattern:** `{matched_pattern}`")

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