import gradio as gr
import joblib
import numpy as np

# ===============================
# LOAD TRAINED MODELS (LOCAL)
# ===============================
lin_model = joblib.load("temp_model.pkl")
log_model = joblib.load("rain_model.pkl")

# ===============================
# PREDICTION FUNCTION
# FEATURE ORDER MUST MATCH TRAINING
# ===============================
def predict_weather(humidity, wind, precipitation, pressure, uv, visibility):
    X = np.array([[humidity, wind, precipitation, pressure, uv, visibility]])

    # Linear Regression → Temperature
    temperature = lin_model.predict(X)[0]

    # Logistic Regression → Rain
    rain_pred = log_model.predict(X)[0]
    rain_result = "Rain" if rain_pred == 1 else "No Rain"

    return float(temperature), rain_result

# ===============================
# GRADIO INTERFACE
# ===============================
interface = gr.Interface(
    fn=predict_weather,
    inputs=[
        gr.Number(label="Humidity (%)"),
        gr.Number(label="Wind Speed"),
        gr.Number(label="Precipitation (%)"),
        gr.Number(label="Atmospheric Pressure"),
        gr.Number(label="UV Index"),
        gr.Number(label="Visibility (km)")
    ],
    outputs=[
        gr.Number(label="Predicted Temperature"),
        gr.Text(label="Rain Prediction")
    ],
    title="🌦️ Weather Prediction App",
    description="Predict Temperature (Linear Regression) and Rain (Logistic Regression) using weather inputs."
)

# ===============================
# LAUNCH APP
# ===============================
interface.launch()
