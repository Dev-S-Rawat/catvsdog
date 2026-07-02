import streamlit as st
import requests
from PIL import Image
import io
import os
from dotenv import load_dotenv

# Load environment variables (to find our API URL)
load_dotenv()

# --- 1. CONFIGURATION ---
# Replace this with your actual live Render API URL we generated!
# It should look something like: "https://catvsdog-rfzr.onrender.com"
API_URL = os.getenv("RENDER_API_URL")

# Basic page setup for a clean, professional look
st.set_page_config(
    page_title="Pet Classifier Dashboard",
    page_icon="🐾",
    layout="centered"
)

# --- 2. THEMING & STYLING ---
# We use custom CSS to force a clean white background and minimalist typography
st.markdown("""
    <style>
    .stApp {
        background-color: white;
        color: #262730;
    }
    h1 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        color: #1A1C1E;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .status-text {
        color: #6D7278;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div[data-testid="stFileUploader"] > label {
        display: none;
    }
    div[data-testid="stFileUploader"] button {
        background-color: white !important;
        color: #262730 !important;
        border: 1px solid #d3d3d3 !important; /* Optional: adds a subtle border so it doesn't blend completely into your white background */
    }
    div[data-testid="stFileUploader"] button:hover {
        background-color: #f8f9fa !important;
        border-color: #a0a0a0 !important;
    }
    .uploadedFile {
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. DASHBOARD UI ---
st.markdown("<h1>🐾 Pet Species AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p class='status-text'>Upload an image of your pet to see the neural network analyze it live. Cat or Dog Only.</p>",
            unsafe_allow_html=True)

# Create the styled file upload zone
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="pet_uploader")

# Create two columns for the analysis section (initially hidden)
col1, col2 = st.columns([1.5, 1])

# If the user uploads an image:
if uploaded_file is not None:
    # --- STEP A: PROCESS LOCAL IMAGE ---
    # Display the uploaded image in the first column
    image = Image.open(uploaded_file)
    with col1:
        st.markdown("<h3 style='text-align: center;'>Analyzed Image</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    # Convert the Pillow image back to bytes so we can send it over the web
    img_bytes = io.BytesIO()
    image.save(img_bytes, format=image.format if image.format else "JPEG")
    img_bytes.seek(0)

    # --- STEP B: CALL THE LIVE API ---
    with col2:
        st.markdown("<h3 style='text-align: center;'>AI Prediction</h3>", unsafe_allow_html=True)

        if not API_URL:
            st.error("Error: RENDER_API_URL environment variable is not set.")
            st.stop()

        with st.spinner('Connecting to neural network...'):
            try:
                # We send the request exactly like our 'curl' command did
                files = {"file": (uploaded_file.name, img_bytes, uploaded_file.type)}
                response = requests.post(f"{API_URL}/predict", files=files)
                response.raise_for_status()  # Raise error if connection fails

                # --- STEP C: DISPLAY THE RESULTS ---
                result = response.json()

                prediction = result["prediction"]
                confidence = result["confidence"]

                # Design Choice: Large prediction results with a clean color indicator
                if prediction == "Dog":
                    st.markdown(
                        f"<h1 style='color: #2E7D32; font-size: 4rem; text-align: center; margin: 0;'>🐕 {prediction}</h1>",
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f"<h1 style='color: #1976D2; font-size: 4rem; text-align: center; margin: 0;'>🐈 {prediction}</h1>",
                        unsafe_allow_html=True)

                # Use Streamlit's progress bar as a confidence gauge
                st.markdown(
                    f"<p style='text-align: center; color: #6D7278; font-size: 1.1rem; margin-top: 10px;'>Confidence Score</p>",
                    unsafe_allow_html=True)
                st.progress(float(confidence))
                st.markdown(f"<h3 style='text-align: center; margin: 0;'>{confidence * 100:.1f}%</h3>",
                            unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Deployment API Error: {str(e)}")
                st.info("Ensure your Render Web Service is 'Live' and the URL in your .env is correct.")

else:
    # If no file is uploaded, show a friendly placeholder image
    with col1:
        st.image("https://placehold.co/600x400?text=Upload+an+Image+to+Start", use_container_width=True)