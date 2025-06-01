# Greenhouse Intelligence System: Crop Climate Feasibility & Management Outlook

##  About the System - Brief Introduction

Greenhouse Intelligence is a Streamlit web application designed to assist users in assessing the climate feasibility for various crops and providing near-future temperature predictions with management suggestions. Users can select a crop, input recent temperature trends, and the application will leverage a Gated Recurrent Unit (GRU) model to predict the next period's temperature. Based on this prediction, it offers insights into climate suitability and recommends potential actuator settings for greenhouse management.

##  Features

* **Interactive Crop Selection:** Users can visually select from a range of available crops.
* **Temperature Trend Input:** Allows input of the last 6 temperature readings to establish a trend.
* **GRU-Powered Prediction:** Utilizes a pre-trained TensorFlow/Keras GRU model to forecast the next temperature period.
* **Climate Feasibility Assessment:** Compares the predicted temperature against the ideal range for the selected crop.
* **Actuator Suggestions:** Recommends settings for fan, watering pump, and water pump actuators based on the predicted temperature.
* **Custom User Interface:** Features a custom background image, styled crop selection cards, and themed alert boxes for an enhanced user experience.

##  Technologies Used

* **Python 3.x**
* **Streamlit:** For building the interactive web application.
* **TensorFlow / Keras:** For loading and running the GRU model (`greenhouse_model.keras`).
* **Scikit-learn:** For the data scaler (`scaler.pkl`).
* **NumPy:** For numerical operations.
* **Pillow (PIL):** For handling images within the application.
* **CSS:** For custom styling elements.

##  Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd GREENHOUSE-APP
    ```

2.  **Create and Activate a conda environement (Recommended):**
    ```bash
    conda activate greenhouse
    ```

3.  **Install Dependencies:**
    Ensure you have a `requirements.txt` file in your project root. If not, you can create one from your active environment where the project works using:
    ```bash
    pip freeze > requirements.txt
    ```
    Then, install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    Key packages that should be in `requirements.txt` include:
    * `streamlit`
    * `numpy`
    * `tensorflow`
    * `scikit-learn`
    * `Pillow`

4.  **Ensure Model and Asset Files are Present:**
    * `greenhouse_model.keras`: The trained GRU model file.
    * `scaler.pkl`: The pre-fitted scaler.
    * `.streamlit/config.toml`: The theme configuration file.
    * Image files for crops (e.g., `tomato_image.jpg`, `bell_pepper_image.jpg`, etc.) in the root directory.
    * Background image (e.g., `dl.jpg`) in the root directory.

## 🚀 How to Run

Once the setup is complete, you can run the Streamlit application using the following command in your terminal (from the `GREENHOUSE-APP` root directory):

```bash
python -m streamlit run app.py
