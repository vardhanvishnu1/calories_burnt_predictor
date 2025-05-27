import streamlit as st
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os


st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="centered", 
    initial_sidebar_state="auto"
)

MODEL_PATH = "calorie_predictor_model.h5"
SCALER_PATH = "scaler.pkl"
LOGO_PATH = os.path.join("static", "images", "logo.jpg")
BG_IMAGE_PATH = os.path.join("static", "images", "bg.jpg")


@st.cache_resource
def load_ml_assets():
    """Loads the Keras model and StandardScaler."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"Error: Model file not found at {MODEL_PATH}. Please ensure it's in the correct directory.")
            st.stop()

        if not os.path.exists(SCALER_PATH):
            st.error(f"Error: Scaler file not found at {SCALER_PATH}. Please ensure it's in the correct directory.")
            st.stop()

        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        
        return model, scaler
    except Exception as e:
        st.error(f"An error occurred while loading ML assets: {e}")
        st.stop()

model, scaler = load_ml_assets()


def inject_custom_css(bg_image_path):
    if os.path.exists(bg_image_path):
        background_image_url = f"url(/static/images/bg.jpg)"
    else:
        background_image_url = "none"

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"] {{
            margin: 0 !important;
            padding: 0 !important;
            width: 100%;
            height: 100%;
            overflow-x: hidden; /* Prevent horizontal scrollbar */
        }}
        .stApp {{
            background-image: {background_image_url};
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            font-family: 'Inter', sans-serif;
            color: #E0E0E0; /* Light text for dark background */
        }}

        /* Target the main content area (the block that holds your app) */
        /* Removed flex properties here as they can interfere with layout="centered" */
        [data-testid="stAppViewContainer"] > .main {{
            background-color: transparent;
            border-radius: 0;
            box-shadow: none;
            padding-top: 2rem; /* Add some top padding */
            padding-bottom: 2rem; /* Add some bottom padding */
            /* No explicit horizontal margins here, let block-container handle centering */
        }}

        /* This is the key for centering the content within the main area */
        .block-container {{
            padding-top: 0rem; /* Remove default top padding */
            padding-bottom: 0rem; /* Remove default bottom padding */
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 700px; /* Set a max-width for readability */
            margin: auto; /* This centers the block-container horizontally within its parent */
        }}

        /* General text color adjustments for visibility on dark background */
        .stTitle {{
            color: #FFFFFF; /* White for titles */
            text-align: center;
        }}
        .stMarkdown p {{
            color: #C0C0C0; /* Slightly less bright for description text */
            text-align: center;
        }}
        h1, h2 {{ /* Targets headings in general */
            color: #4CAF50; /* Green for titles/headings to stand out */
            text-align: center;
            font-weight: 700;
        }}
        label {{
            color: #F0F0F0; /* Label text color for input fields */
        }}
        /* Styling for Streamlit's input components */
        .stNumberInput, .stSelectbox {{
            color: #F0F0F0;
            background-color: rgba(30, 30, 30, 0.7); /* Slightly translucent dark background for inputs */
            border: 1px solid rgba(70, 70, 70, 0.7);
            border-radius: 0.5rem;
            margin-bottom: 1rem; /* Add some space between inputs */
        }}
        /* Adjusting the actual input fields within number/select boxes for transparent background */
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > button {{
            background-color: transparent !important;
            color: #F0F0F0 !important;
        }}
        /* Plus/minus buttons for number input */
        .stNumberInput > div > div > button {{
            background-color: rgba(50, 50, 50, 0.5) !important;
            color: #F0F0F0 !important;
            border: none !important;
        }}
        /* Dropdown arrow in selectbox */
        .stSelectbox > div > div > div {{
            background-color: transparent !important;
            color: #F0F0F0 !important;
        }}
        /* Options list in dropdown when opened */
        [data-baseweb="popover"] {{
             background-color: #333 !important;
             color: #F0F0F0 !important;
        }}
        [data-baseweb="menu"] li {{
            background-color: #333 !important;
            color: #F0F0F0 !important;
        }}
        [data-baseweb="menu"] li:hover {{
            background-color: #555 !important;
        }}


        .stButton>button {{
            width: 100%;
            border-radius: 0.5rem;
            font-size: 1.1rem;
            padding: 0.75rem 1rem;
            background-color: #0d6efd; /* Blue for button */
            color: white;
            border: none;
            transition: background-color 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #0b5ed7;
        }}
        .stAlert {{
            border-radius: 0.5rem;
            margin-top: 1.5rem;
            background-color: rgba(76, 175, 80, 0.7); /* Semi-transparent success green */
            border-left: 5px solid #4CAF50;
            color: white; /* Alert text color */
        }}
        .stAlert.error {{
            background-color: rgba(244, 67, 54, 0.7); /* Semi-transparent error red */
            border-left-color: #F44336;
            color: white; /* Alert text color */
        }}

        /* Specific form container - ensure it's transparent and has no shadow */
        .st-emotion-cache-1j04qpe {{ /* This class targets the form. Adjust if needed. */
             background-color: transparent;
             padding: 0;
             border-radius: 0;
             box-shadow: none;
             margin-top: 2rem;
             margin-bottom: 2rem; /* Add margin to separate from other elements */
        }}

        /* Remove default Streamlit header/footer padding for tighter design */
        header, footer {{
            display: none !important;
        }}

    </style>
    """, unsafe_allow_html=True)

inject_custom_css(BG_IMAGE_PATH)



col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=300, use_column_width=False)
    else:
        st.warning(f"Logo not found at {LOGO_PATH}. Displaying placeholder.")
        st.markdown("<p style='text-align: center;'>App Logo</p>", unsafe_allow_html=True)


st.title("Calories Burnt Predictor")
st.markdown("Enter your workout details to estimate calories burned")

st.markdown("---")

# --- Input Form ---
with st.form(key='calories_prediction_form', clear_on_submit=False):
    age = st.number_input("Age", min_value=1, max_value=120, value=25, help="Your age in years.")
    gender_options = {"Male": 0, "Female": 1}
    gender_display = st.selectbox("Gender", list(gender_options.keys()), help="Your biological gender.")
    gender = gender_options[gender_display]

    height = st.number_input("Height (cm)", min_value=10, max_value=250, value=170, help="Your height in centimeters.")
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0, help="Your weight in kilograms.")
    duration = st.number_input("Duration (minutes)", min_value=1, max_value=600, value=30, help="Duration of your exercise in minutes.")
    heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=220, value=120, help="Average heart rate during exercise.")
    body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=42.0, value=37.0, step=0.1, format="%.1f", help="Your body temperature after exercise in Celsius.")

    submitted = st.form_submit_button("Predict Calories Burnt")

    if submitted:
        try:
            input_data = np.array([
                age, gender, height, weight, duration, heart_rate, body_temp
            ]).reshape(1, -1)

            scaled_input_data = scaler.transform(input_data)

            raw_prediction = model.predict(scaled_input_data)
            prediction = round(float(raw_prediction[0][0]), 2)

            st.success(f"## 🔥 Estimated Calories Burnt: **{prediction}** kcal")

        except ValueError:
            st.error("Please ensure all input fields are filled with valid numerical values.")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

st.markdown("---")
