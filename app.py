import os
import gradio as gr
import joblib

# Load the trained Decision Tree model at startup
deployed_dt = joblib.load('diabetes_prediction_model.pkl')

# --- CODE BLOCK: PREDICTION LOGIC FOR 5 FEATURES ---
def predict_diabetes(Pregnancies, PlasmaGlucose, DiastolicBloodPressure,TricepsThickness, SerumInsulin, BMI, Age):
    # The model expects a 2D array matching the exact order of x_train
    input_data = [[Pregnancies, PlasmaGlucose, DiastolicBloodPressure,TricepsThickness, SerumInsulin, BMI, Age]]
    prediction = deployed_dt.predict(input_data)
    
    # Interpret the binary outcome (typically 1 for positive, 0 for negative)
    if prediction[0] == 1:
        return "Prediction: High Risk of Diabetes (Positive)"
    else:
        return "Prediction: Low Risk of Diabetes (Negative)"
# ---------------------------------------------------

# --- CODE BLOCK: GRADIO INTERFACE SETUP ---
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Plasma Glucose (mg/dL)"),
        gr.Number(label="Diastolic Blood Pressure (mm Hg)"),
        gr.Number(label="Triceps Skin Fold Thickness (mm)"),
        gr.Number(label="Serum Insulin (mu U/mL)"),
        gr.Number(label="Body Mass Index (kg/m^2)"),
        gr.Number(label="Age (Years)")
    ],
    outputs=gr.Text(label="Assessment Result"),
    title="Diabetes Prediction System",
    description="Enter the medical metrics to predict diabetes risk using a Decision Tree Machine Learning model."
)
# ------------------------------------------

if _name_ == "_main_":
    # Render network configuration
    interface.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
