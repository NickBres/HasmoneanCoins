import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
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
ROBOFLOW_API_KEY = "tBK1cI9pNviSzUq3wQ4N"

# App Header
st.title("Hasmonean Coins Recognition Model")
st.markdown("### Analyze Hasmonean coin inscriptions to identify rulers based on detected letters.")

# Load patterns
patterns = load_patterns()

# ---- Sample Demonstration Coins ----
st.header("Choose a Sample Coin or Upload Your Own")

# Preloaded sample images (store in the `demo_coins/` folder)
SAMPLE_COINS = {
    "Coin 1 - Alexander Jannaeus": "demo_coins/alexander_jannaeus.png",
    "Coin 2 - John Hyrcanus": "demo_coins/john_hyrcanus.png",
    "Coin 3 - Yehuda Aristobelus": "demo_coins/yehuda_aristobelus.png"
}

# Dropdown for selecting a sample coin
selected_coin = st.selectbox("Select a Sample Coin:", ["Upload New"] + list(SAMPLE_COINS.keys()))

# Hide file uploader when a sample coin is chosen
if selected_coin == "Upload New":
    uploaded_file = st.file_uploader("Or upload your own image", type=["jpg", "png"])
else:
    uploaded_file = None

# Load the selected sample coin or the uploaded file
image = None
image_bytes = None
if selected_coin != "Upload New":
    image_path = SAMPLE_COINS[selected_coin]
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image = Image.open(image_path)
    st.image(image, caption=f"Selected Sample: {selected_coin}", use_container_width=True)
elif uploaded_file:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ---- Process Image If Available ----
if image:
    # ---- Confidence Threshold ----
    st.header("Confidence Filter")
    st.write("Adjust the confidence level. Only letters detected with confidence above this threshold will be displayed.")
    confidence_threshold = st.slider("Minimum Confidence", 1, 100, 50)

    # Send image to Roboflow API
    api_url = f"https://detect.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}?api_key={ROBOFLOW_API_KEY}"
    response = requests.post(api_url, files={"file": image_bytes})

    if response.status_code == 200:
        result = response.json()
        predictions = result.get("predictions", [])

        # Save JSON result (only for uploaded images, not demo images)
        if uploaded_file:
            os.makedirs("inference_results", exist_ok=True)
            json_filename = f"inference_results/{uploaded_file.name.split('.')[0]}.json"
            with open(json_filename, "w") as json_file:
                json.dump(result, json_file, indent=4)

        # ---- Filter Predictions ----
        filtered_predictions = [p for p in predictions if p["confidence"] * 100 >= confidence_threshold]

        # ---- Letter Selection ----
        st.header("Select Letters to Display")
        found_letters = list(set(p["class"] for p in filtered_predictions))
        letter_visibility = {letter: st.checkbox(letter, value=True) for letter in found_letters}

        # ---- Sort Letters & Apply Filters ----
        sorted_predictions = sort_letters(filtered_predictions)
        visible_predictions = [p for p in sorted_predictions if letter_visibility.get(p["class"], False)]

        # ---- Display Recognized Letters ----
        st.header("Recognition Demonstration")
        st.write("The image below shows the detected letters after filtering.")
        font_size = st.slider("Choose text size", min_value=5, max_value=30, value=15)  # User picks size

        # Use user-selected font size when visualizing letters
        detected_image = visualize_detections(image.copy(), visible_predictions, font_size=font_size)
        st.image(detected_image, caption="Detected Letters", use_container_width=True)

        # ---- Collapsible JSON Data ----
        with st.expander("View Raw JSON Data"):
            st.json(result)

        # ---- Ruler Prediction Section ----
        st.header("Predicted Ruler")
        ruler_name, matched_letters = match_ruler_name(visible_predictions, patterns)
        st.subheader(f"**{ruler_name}**")

        # ---- Visualizing the Matched Pattern ----
        if ruler_name != "Unknown Ruler":
            st.write("The image below highlights only the letters used in the ruler's pattern.")
            ruler_match_image = visualize_detections(image.copy(), matched_letters, font_size=font_size)
            st.image(ruler_match_image, caption=f"Matched Pattern for {ruler_name}", use_container_width=True)

        # ---- Pattern Management Section ----
        st.header("Pattern Settings")
        new_ruler = st.text_input("Enter Ruler Name:")
        new_pattern = st.text_input("Enter Pattern (use `*` for any letter, e.g., 'Aleph-*-Daled-Resh')")

        if st.button("Save Pattern"):
            if new_ruler and new_pattern:
                save_pattern(new_ruler, new_pattern)
                st.success(f"Saved pattern for {new_ruler}: `{new_pattern}`")
                st.rerun()

        st.subheader("Manage Existing Patterns")

        for ruler, pattern_list in patterns.items():
            st.write(f"**{ruler}:**")
            for index, pattern in enumerate(pattern_list):
                col1, col2 = st.columns([4, 1])
                col1.write(f"- `{pattern}`")
                safe_key = generate_safe_key(ruler, pattern, index)
                if col2.button("❌", key=safe_key):
                    delete_pattern(ruler, pattern)
                    st.rerun()

        # Show stored patterns
        with st.expander("View Saved Patterns"):
            st.json(patterns)

    else:
        st.error("Error: Could not process the image. Check your API key and model settings.")