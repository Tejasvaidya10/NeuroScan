import streamlit as st
from PIL import Image
import numpy as np
import json
import sys
import os

# Add the root directory to the python path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import preprocess_image, load_trained_model

# Page Configuration
st.set_page_config(
    page_title="NeuroScan",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #2b313e;
        color: white;
        border: 1px solid #4a4e69;
    }
    .stButton>button:hover {
        background-color: #4a4e69;
        color: #ffffff;
    }
    .reportview-container .markdown-text-container {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    h1 {
        font-weight: 300;
        color: #f0f2f6;
    }
    h3 {
        color: #c9d1d9;
    }
    </style>
    """, unsafe_allow_html=True)

DEFAULT_CLASSES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']


def load_class_labels(model_dir):
    """Load class labels from JSON file saved during training."""
    labels_path = os.path.join(model_dir, 'class_labels.json')
    if os.path.exists(labels_path):
        with open(labels_path, 'r') as f:
            return json.load(f)
    return DEFAULT_CLASSES


def main():
    st.title("NeuroScan")
    st.write("Upload an MRI scan to detect the presence and type of brain tumor.")

    # Sidebar
    st.sidebar.title("Configuration")
    model_path = st.sidebar.text_input("Model Path", "models/brain_tumor_model.h5")

    # Load Model
    model = load_trained_model(model_path)
    if model is None:
        st.warning(f"Model not found at {model_path}. Please train the model first or check the path.")
    else:
        st.sidebar.success("Model loaded successfully.")

    # Load class labels from the same directory as the model
    model_dir = os.path.dirname(model_path)
    classes = load_class_labels(model_dir)

    # File Uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded MRI Scan', use_column_width=True)
        except Exception:
            st.error("Could not open the uploaded file. Please upload a valid image (JPG, PNG, JPEG).")
            return

        # Predict Button
        if st.button('Analyze Scan'):
            if model is None:
                st.error("Cannot predict: Model is not loaded.")
            else:
                with st.spinner('Analyzing...'):
                    try:
                        processed_image = preprocess_image(image)
                        prediction = model.predict(processed_image)
                    except Exception as e:
                        st.error(f"Error during analysis: {e}")
                        return

                    class_idx = np.argmax(prediction)
                    confidence = np.max(prediction) * 100

                    result_class = classes[class_idx]

                    # Display Result
                    st.divider()
                    st.header("Results")
                    st.success(f"Prediction: {result_class}")
                    st.info(f"Confidence: {confidence:.2f}%")

                    if confidence < 70:
                        st.warning("Low confidence prediction. Results may not be reliable.")

                    # Detailed probabilities
                    st.write("Detailed Probabilities:")
                    for i, class_name in enumerate(classes):
                        st.progress(int(prediction[0][i] * 100))
                        st.caption(f"{class_name}: {prediction[0][i]*100:.2f}%")

if __name__ == "__main__":
    main()
