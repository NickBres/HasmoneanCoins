import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import json
import base64
import os
from inference_sdk import InferenceHTTPClient, InferenceConfiguration  # ✅ Import Roboflow SDK
from utils import (
    match_ruler_name, load_patterns, save_pattern,
    visualize_detections, delete_pattern, generate_safe_key, sort_letters_hebrew, extract_feature_vector,
    normalize_feature_vector
)

import pickle
import numpy as np
import pandas as pd

# Load the trained XGBoost model
MODEL_PATH = "xgboost_ruler_classifier.pkl"
with open(MODEL_PATH, "rb") as f:
    ruler_classifier = pickle.load(f)

# API Details
ROBOFLOW_MODEL = "hasmonean_coins_letter_detection"
ROBOFLOW_VERSION = "15"
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")

if not ROBOFLOW_API_KEY:
    raise ValueError("Missing API Key! Set the ROBOFLOW_API_KEY environment variable.")

# Initialize Roboflow Client
CLIENT = InferenceHTTPClient("https://detect.roboflow.com", ROBOFLOW_API_KEY)

# Set Confidence and Overlay Defaults
CONFIDENCE_THRESHOLD = 0.35  # Default 35%
OVERLAY_THRESHOLD = 0.70  # Default 70%

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

# Load stored ruler patterns
patterns = load_patterns()

# ---- Upload Section ----
uploaded_file = st.file_uploader("📂 Choose an image (JPG/PNG)", type=["jpg", "png"])

if not uploaded_file:
    st.info("Please upload an image to continue.")
    st.stop()

# Read the uploaded image
image_bytes = uploaded_file.getvalue()
image = Image.open(io.BytesIO(image_bytes))
image_base64 = base64.b64encode(image_bytes).decode("utf-8")

# ---- Track Image Changes to Preserve Selection ----
if "previous_image" not in st.session_state:
    st.session_state.previous_image = None

image_hash = hash(image_bytes)  # Unique identifier for the uploaded image
if st.session_state.previous_image != image_hash:
    st.session_state.selected_letters = None  # Reset selection only if a new image is uploaded
    st.session_state.previous_image = image_hash  # Store new image hash

# ---- Send Image to Roboflow API ----
with CLIENT.use_configuration(custom_configuration):
    result = CLIENT.infer(image_base64, model_id=f"{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}")

# Check API Response
if not result or "predictions" not in result:
    st.error("❌ Error: Could not process the image. Check your API key and model settings.")
    st.stop()

predictions = result["predictions"]

with st.expander("🔠 Adjust Recognized Letters (Optional)"):
    st.write("Uncheck incorrect letters to refine the results.")

    if predictions:
        sorted_predictions = sort_letters_hebrew(predictions, image.width, image.height)

        # Initialize selection state only if it's not already set
        if st.session_state.selected_letters is None:
            st.session_state.selected_letters = {f"{p['class']}_{i}" for i, p in enumerate(sorted_predictions)}

        # Check if all letters are selected
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
    else:
        visible_predictions = []

font_size = st.slider("Text Size", min_value=5, max_value=50, value=15)
filtered_image = visualize_detections(image.copy(), visible_predictions, font_size=font_size)
st.image(filtered_image, caption="🔍 Filtered Letters", use_container_width=True)






# ---- Predict Ruler from Letters ----
if visible_predictions:
    # Extract feature vector
    feature_vector = extract_feature_vector({"predictions": visible_predictions})

    print("🔍 Raw Feature Vector:", feature_vector)

    # Normalize feature vector
    normalized_feature_vector = normalize_feature_vector(feature_vector, image.width, image.height)

    print("🔍 Debug After Normalization:", feature_vector)  # Check AFTER normalization

    # Convert normalized dictionary to NumPy array
    normalized_feature_array = np.array([normalized_feature_vector[key] for key in feature_vector.keys()]).reshape(1,
                                                                                                                   -1)

    # Get prediction probabilities using the normalized feature vector
    probabilities = ruler_classifier.predict_proba(normalized_feature_array)[0]

    # Find the most confident class
    predicted_class = np.argmax(probabilities)
    confidence_score = probabilities[predicted_class] * 100  # Convert to percentage

    # Map numeric prediction to full ruler name
    class_mapping = {
        0: "ALEXANDER JANNAEUS",
        1: "JOHN HYRCANUS I",
        2: "YEHUDA ARISTOBULUS"
    }
    predicted_ruler = class_mapping.get(predicted_class, "Unknown")

    st.subheader(f"🤔 I think this is **{predicted_ruler}**")
    st.write(f"I'm about **{confidence_score:.2f}%** sure, but you might want to double-check!")

    # Add warning if confidence is high but misclassification is possible
    st.warning(
        "⚠️ This prediction is made by a machine learning model and **may not be accurate**.\n"
    )

    with st.expander("📊 View Feature Vector Sent to ML Model"):
        st.write(
            "Here are the extracted feature vectors before and after normalization. The normalized version is used for classification.")

        # Define column names for readability
        column_names = [
            "A_Count", "A_AvgX", "A_AvgY", "A_StdX", "A_StdY", "A_BoxRatio",
            "Bet_Count", "Bet_AvgX", "Bet_AvgY", "Bet_StdX", "Bet_StdY", "Bet_BoxRatio",
            "Gimel_Count", "Gimel_AvgX", "Gimel_AvgY", "Gimel_StdX", "Gimel_StdY", "Gimel_BoxRatio",
            "Dalet_Count", "Dalet_AvgX", "Dalet_AvgY", "Dalet_StdX", "Dalet_StdY", "Dalet_BoxRatio",
            "Hei_Count", "Hei_AvgX", "Hei_AvgY", "Hei_StdX", "Hei_StdY", "Hei_BoxRatio",
            "Vav_Count", "Vav_AvgX", "Vav_AvgY", "Vav_StdX", "Vav_StdY", "Vav_BoxRatio",
            "Het_Count", "Het_AvgX", "Het_AvgY", "Het_StdX", "Het_StdY", "Het_BoxRatio",
            "Yod_Count", "Yod_AvgX", "Yod_AvgY", "Yod_StdX", "Yod_StdY", "Yod_BoxRatio",
            "Kaf_Count", "Kaf_AvgX", "Kaf_AvgY", "Kaf_StdX", "Kaf_StdY", "Kaf_BoxRatio",
            "Lamed_Count", "Lamed_AvgX", "Lamed_AvgY", "Lamed_StdX", "Lamed_StdY", "Lamed_BoxRatio",
            "Mem_Count", "Mem_AvgX", "Mem_AvgY", "Mem_StdX", "Mem_StdY", "Mem_BoxRatio",
            "Nun_Count", "Nun_AvgX", "Nun_AvgY", "Nun_StdX", "Nun_StdY", "Nun_BoxRatio",
            "Resh_Count", "Resh_AvgX", "Resh_AvgY", "Resh_StdX", "Resh_StdY", "Resh_BoxRatio",
            "Taf_Count", "Taf_AvgX", "Taf_AvgY", "Taf_StdX", "Taf_StdY", "Taf_BoxRatio"
        ]
        # Convert raw feature vector to DataFrame
        if len(feature_vector) == len(column_names):
            raw_df = pd.DataFrame([feature_vector], columns=column_names)
            st.subheader("🔹 Raw Feature Vector (Before Normalization)")
            st.dataframe(raw_df)

            # Convert normalized feature vector to DataFrame
            normalized_df = pd.DataFrame([normalized_feature_vector], columns=column_names)
            st.subheader("🔹 Normalized Feature Vector (Used for Classification)")
            st.dataframe(normalized_df)
        else:
            st.error("⚠️ Feature vector length does not match expected columns. Check preprocessing.")

else:
    st.info("No letters detected. Please check your selections or upload a clearer image.")


# ---- Collapsible JSON Data ----
with st.expander("📜 View Raw JSON Data"):
    st.json(result)

# ---- Move "How This Works" to a Collapsed Expander ----
with st.expander("ℹ️ How This Works (Click to Expand)"):
    st.markdown("""
    ### 🔍 How This Works
    1️⃣ **Letter Recognition**  
       - We use a pretrained **Roboflow 3.0** model to detect ancient Hasmonean letters in coin images.  
       - The model only considers **letters with at least 35% confidence** and applies a **70% overlay threshold** to reduce errors.  

    2️⃣ **Ruler Classification**  
       - Once letters are detected, we pass this information to a separate **XGBoost model**.  
       - The XGBoost model processes the detected letters and **classifies the coin to one of three rulers**:  
         - 🏛️ **ALEXANDER JANNAEUS**  
         - 🏛️ **JOHN HYRCANUS I**  
         - 🏛️ **YEHUDA ARISTOBULUS**  
    """)