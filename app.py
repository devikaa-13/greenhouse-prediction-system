# app.py (Cleaned up to work with config.toml)

import streamlit as st
import base64
import numpy as np
import tensorflow as tf
import pickle

# --- CROP IDEAL CONDITIONS DATABASE ---
CROP_CONDITIONS = {
    "Tomatoes": {"temp_min": 21, "temp_max": 27, "humidity_min": 60, "humidity_max": 80, "image_file": "tomato_image.jpg"},
    "Bell Peppers": {"temp_min": 21, "temp_max": 29, "humidity_min": 55, "humidity_max": 75, "image_file": "bell_pepper_image.jpg"},
    "Lettuce": {"temp_min": 15, "temp_max": 22, "humidity_min": 50, "humidity_max": 70, "image_file": "lettuce_image.jpg"},
    "Cucumbers": {"temp_min": 20, "temp_max": 30, "humidity_min": 60, "humidity_max": 75, "image_file": "cucumber_image.jpg"},
    "Strawberries": {"temp_min": 15, "temp_max": 25, "humidity_min": 60, "humidity_max": 75, "image_file": "strawberry_image.jpg"},
    "Basil": {"temp_min": 18, "temp_max": 28, "humidity_min": 50, "humidity_max": 70, "image_file": "basil_image.jpg"}
}
AVAILABLE_CROPS = list(CROP_CONDITIONS.keys())

# --- FUNCTION TO SET BACKGROUND AND STYLES ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded_page_bg = base64.b64encode(img_data).decode()
    # These variables are now mostly controlled by config.toml, but we keep them for custom elements
    text_color = "#3A241D" 
    card_bg_color = "#F7F7F7" 
    card_border_color = "#8B4513"
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpeg;base64,{b64_encoded_page_bg});
            background-size: cover; background-repeat: no-repeat;
            background-attachment: fixed; background-position: left bottom;
        }}
        /* The main block container styling is still useful */
        .main .block-container {{
            background-color: rgba(255, 255, 240, 0.9); padding: 2rem;
            border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        
        /* We remove the button styles from here as config.toml now handles them */
        
        .crop-card {{
             border: 3px solid {card_border_color};
             border-radius: 12px; padding: 15px; margin-bottom: 10px;
             background-color: {card_bg_color};
             box-shadow: 3px 3px 7px rgba(0,0,0,0.15); text-align: center;
             height: 180px; 
             display: flex; flex-direction: column; justify-content: center; align-items: center;
        }}
        .crop-card img {{
            max-width: 90%; max-height: 120px; object-fit: contain; 
            border-radius: 6px; 
        }}
        .crop-card p.crop-name {{ 
            margin-top: 10px; font-weight: bold; font-size: 1.1em; 
            color: {text_color} !important; 
        }}
        </style>
        """
    st.markdown(style, unsafe_allow_html=True)


# --- APP CONFIGURATION ---
st.set_page_config(page_title="Greenhouse Intelligence", layout="centered")

# --- SET PAGE BACKGROUND ---
try:
    set_background('dl.jpg') 
except FileNotFoundError:
    st.warning("Page background image 'dl.jpg' not found.")

# --- LOAD MODELS AND SCALER ---
@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path)

@st.cache_resource
def load_pickle_object(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

try:
    temp_model = load_tf_model('greenhouse_model.keras')
    temp_scaler = load_pickle_object('scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# --- HELPER FUNCTION ---
def get_temp_actuator_suggestions(predicted_temperature, selected_crop_name="General", crop_ideal_temp_min=22.0, crop_ideal_temp_max=30.0):
    VERY_HOT_TEMP = crop_ideal_temp_max + 5.0 
    suggestions = {'Fan Actuator': 'OFF', 'Watering Plant Pump': 'OFF', 'Water Pump Actuator': 'OFF'}
    reasoning = ""
    if predicted_temperature > crop_ideal_temp_max: 
        suggestions['Fan Actuator'] = 'ON'
        reasoning = f"Predicted temp ({predicted_temperature:.1f}°C) is above {selected_crop_name}'s ideal max ({crop_ideal_temp_max}°C). Fan ON."
        if predicted_temperature > VERY_HOT_TEMP:
            suggestions['Watering Plant Pump'] = 'ON'
            suggestions['Water Pump Actuator'] = 'ON'
            reasoning += f" Temp is also very high (>{VERY_HOT_TEMP}°C for {selected_crop_name}), activating watering."
    elif predicted_temperature < crop_ideal_temp_min:
        reasoning = f"Predicted temp ({predicted_temperature:.1f}°C) is below {selected_crop_name}'s ideal min ({crop_ideal_temp_min}°C). Consider heating."
    else:
        reasoning = f"Predicted temp ({predicted_temperature:.1f}°C) is within the ideal range for {selected_crop_name}."
    suggestions['Reasoning'] = reasoning
    return suggestions

# --- SESSION STATE HANDLING ---
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = None

# --- STREAMLIT APP LAYOUT ---
st.title("Crop Climate Feasibility & Management Outlook")
st.write("""
Select a crop below, then enter the current temperature trend 
to predict the near-future greenhouse climate and assess its suitability.
""")
st.markdown("---")

# --- CROP SELECTION ---
st.subheader("Select a Crop to Analyze:")
# We still use st.button here, but its styling will now come from config.toml
num_columns = 3
cols = st.columns(num_columns)
col_idx = 0
for crop_name in AVAILABLE_CROPS:
    with cols[col_idx]: 
        crop_data = CROP_CONDITIONS[crop_name]
        image_path = crop_data.get("image_file")
        
        try:
            with open(image_path, "rb") as img_f:
                b64_img = base64.b64encode(img_f.read()).decode()
            card_html = f"""
            <div class="crop-card">
                <img src="data:image/jpeg;base64,{b64_img}" alt="{crop_name}">
                
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        except FileNotFoundError:
            st.error(f"Image not found: {image_path}")
        
        # The user will click this button.
        # It will be styled as a SECONDARY button by default.
        if st.button(f"{crop_name}", key=f"select_{crop_name}"):
            st.session_state.selected_crop = crop_name
            st.rerun()

    col_idx = (col_idx + 1) % num_columns


# --- FEASIBILITY CHECK (TEMPERATURE PREDICTION PART) ---
if st.session_state.selected_crop:
    selected_crop = st.session_state.selected_crop
    
    st.markdown("---") 
    st.subheader(f"Enter Temperature Trend for: {selected_crop}")
    
    with st.form(key="feasibility_form"):
        user_input_lookback_temps = st.text_input("Last 6 temperature readings (separated by commas):", 
                                                placeholder="e.g., 28.5, 29, 29.5, 30, 30.5, 31")
        
        # This button is the PRIMARY button, and its background is now set to white by config.toml
        submitted = st.form_submit_button("Check Climate Feasibility")

    if submitted:
        if not user_input_lookback_temps:
            st.warning("Please enter recent temperature data.")
        else:
            try:
                look_back = 6 
                temp_strings = user_input_lookback_temps.split(',')
                input_temps = [float(temp.strip()) for temp in temp_strings]
                if len(input_temps) == look_back:
                    input_data_gru = np.array(input_temps).reshape(-1, 1)
                    scaled_input_gru = temp_scaler.transform(input_data_gru)
                    reshaped_input_gru = np.reshape(scaled_input_gru, (1, look_back, 1))
                    
                    with st.spinner("Predicting..."):
                        predicted_scaled_gru = temp_model.predict(reshaped_input_gru)
                    
                    predicted_temp_gru = temp_scaler.inverse_transform(predicted_scaled_gru)[0][0]
                    st.subheader(f"Climate Outlook for: {selected_crop}")
                    st.success(f"Predicted temperature for the next period: **{predicted_temp_gru:.2f}°C**")
                    
                    crop_info = CROP_CONDITIONS.get(selected_crop)
                    if crop_info:
                        ideal_min = crop_info["temp_min"]
                        ideal_max = crop_info["temp_max"]
                        st.write(f"Ideal temperature range for **{selected_crop}**: **{ideal_min}°C - {ideal_max}°C**")
                        
                        feasibility_message = ""
                        buffer = 2.0
                        if predicted_temp_gru >= ideal_min and predicted_temp_gru <= ideal_max:
                            feasibility_message = f"✅ Conditions are **within ideal range** for {selected_crop}."
                        elif predicted_temp_gru > ideal_max and predicted_temp_gru <= ideal_max + buffer:
                            feasibility_message = f"⚠️ Temp is **slightly above** ideal for {selected_crop}."
                        elif predicted_temp_gru < ideal_min and predicted_temp_gru >= ideal_min - buffer:
                            feasibility_message = f"⚠️ Temp is **slightly below** ideal for {selected_crop}."
                        elif predicted_temp_gru > ideal_max + buffer:
                            feasibility_message = f"❌ **Warning:** Temp is **significantly above** ideal for {selected_crop}."
                        elif predicted_temp_gru < ideal_min - buffer:
                            feasibility_message = f"❌ **Warning:** Temp is **significantly below** ideal for {selected_crop}."
                        st.markdown(f"**Feasibility Statement:** {feasibility_message}")
                        actuator_suggestions = get_temp_actuator_suggestions(predicted_temp_gru, selected_crop, ideal_min, ideal_max)
                        st.info(f"Recommended Actuator Settings for {selected_crop}:")
                        col1_act, col2_act, col3_act = st.columns(3)
                        col1_act.metric("Fan Actuator", actuator_suggestions['Fan Actuator'])
                        col2_act.metric("Watering Pump", actuator_suggestions['Watering Plant Pump'])
                        col3_act.metric("Water Pump", actuator_suggestions['Water Pump Actuator'])
                        st.caption(f"Actuator Reasoning: {actuator_suggestions.get('Reasoning', '')}")
                else:
                    st.error(f"Please enter exactly {look_back} temperature readings.")
            except ValueError:
                st.error("Invalid temperature input. Please ensure you enter numbers separated by commas.")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
else: 
    st.info("Please select a crop to proceed with the feasibility check.")
