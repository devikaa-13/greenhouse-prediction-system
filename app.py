# app.py
# Final Polished Version: All functionality + Visible placeholder text.

import streamlit as st
import base64
import numpy as np
import tensorflow as tf
import pickle

# --- FUNCTION TO SET BACKGROUND AND STYLES ---
def set_background(image_file):
    """
    This function sets a background image and injects CSS for styling.
    """
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    text_color = "#000000" # Black text
    
    # CSS to inject
    style = f"""
        <style>
        /* Main background settings */
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: left bottom;
        }}

        /* Make the main content block semi-transparent white for readability */
        .main .block-container {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}

        /* Style all text elements to be black */
        h1, h2, h3, h4, h5, h6, p, li, label, .st-emotion-cache-16idsys p, .st-emotion-cache-1xarl3l p {{
             color: {text_color} !important;
        }}
        
        /* Make instructional text bigger */
        .st-emotion-cache-16idsys p {{
            font-size: 1.1rem !important;
        }}

        /* Make the text input box white with black text */
        div.stTextInput>div>div>input {{
            color: {text_color};
            background-color: #FFFFFF;
        }}
        
        /* --- FINAL FIX: Make the placeholder text visible --- */
        div.stTextInput>div>div>input::placeholder {{
            color: #888888 !important; /* A visible grey color */
            opacity: 1 !important;
        }}
        /* For older browsers (optional but good practice) */
        div.stTextInput>div>div>input::-webkit-input-placeholder {{
            color: #888888 !important;
        }}
        div.stTextInput>div>div>input:-ms-input-placeholder {{
            color: #888888 !important;
        }}


        /* Style the "Predict" button */
        .stButton>button {{
            color: #FFFFFF;
            background-color: #0068C9;
            border: none;
        }}
        .stButton>button:hover {{
            background-color: #00509E;
            color: #FFFFFF;
        }}

        /* Style the metric boxes for results */
        div.stMetric-value, div.stMetric-delta {{
            color: {text_color} !important;
        }}
        div.stMetric-label {{
            color: rgba(0,0,0,0.7) !important;
        }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)


# --- APP CONFIGURATION ---
st.set_page_config(page_title="Greenhouse Control", layout="centered")

# --- SET BACKGROUND ---
try:
    set_background('dl.jpg') 
except FileNotFoundError:
    st.warning("Background image 'dl.jpg' not found. Please make sure it is in the same folder.")


# --- LOAD THE SAVED MODEL AND SCALER ---
try:
    model = tf.keras.models.load_model('greenhouse_model.keras')
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.error("Please make sure 'greenhouse_model.keras' and 'scaler.pkl' are in the same folder as app.py.")
    st.stop()


# --- HELPER FUNCTION FOR SUGGESTIONS ---
def get_actuator_suggestions(predicted_temperature):
    IDEAL_TEMP_MAX = 30.0
    IDEAL_TEMP_MIN = 22.0
    VERY_HOT_TEMP = 35.0

    suggestions = {
        'Fan Actuator': 'OFF',
        'Watering Plant Pump': 'OFF',
        'Water Pump Actuator': 'OFF'
    }

    if predicted_temperature > IDEAL_TEMP_MAX:
        suggestions['Fan Actuator'] = 'ON'
    if predicted_temperature > VERY_HOT_TEMP:
        suggestions['Watering Plant Pump'] = 'ON'
        suggestions['Water Pump Actuator'] = 'ON'
        
    return suggestions


# --- STREAMLIT USER INTERFACE ---
st.title("GREENHOUSE PREDICTIVE CONTROL SYSTEM")

st.write("""
Enter the last 6 temperature readings to predict the next temperature and get actuator suggestions.
""")

# User input
user_input = st.text_input("Enter 6 temperatures separated by commas", label_visibility="collapsed", placeholder="e.g., 28.5, 29, 29.5, 30, 30.5, 31")

if st.button("Predict"):
    if user_input:
        try:
            look_back = 6
            temp_strings = user_input.split(',')
            input_temps = [float(temp.strip()) for temp in temp_strings]

            if len(input_temps) == look_back:
                input_data = np.array(input_temps).reshape(-1, 1)
                scaled_input = scaler.transform(input_data)
                reshaped_input = np.reshape(scaled_input, (1, look_back, 1))

                predicted_scaled = model.predict(reshaped_input)
                predicted_temp = scaler.inverse_transform(predicted_scaled)[0][0]

                suggestions = get_actuator_suggestions(predicted_temp)

                st.success(f"Predicted Next Temperature: {predicted_temp:.2f} °C")
                st.info("Recommended Actuator Settings:")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Fan Actuator", suggestions['Fan Actuator'])
                col2.metric("Watering Plant Pump", suggestions['Watering Plant Pump'])
                col3.metric("Water Pump Actuator", suggestions['Water Pump Actuator'])
            else:
                st.error(f"Please enter exactly {look_back} temperature readings.")
        except ValueError:
            st.error("Invalid input. Please ensure you enter numbers separated by commas.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.warning("Please enter temperature data.")