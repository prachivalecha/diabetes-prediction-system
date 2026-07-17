import os
import gradio as gr
import joblib

# Load the trained model
deployed_dt = joblib.load("diabetes_prediction_model.pkl")


# Prediction Function
def predict_diabetes(
    Pregnancies,
    PlasmaGlucose,
    DiastolicBloodPressure,
    TricepsThickness,
    SerumInsulin,
    BMI,
    DiabetesPedigree,
    Age
):
    input_data = [[
        Pregnancies,
        PlasmaGlucose,
        DiastolicBloodPressure,
        TricepsThickness,
        SerumInsulin,
        BMI,
        DiabetesPedigree,
        Age
    ]]

    prediction = deployed_dt.predict(input_data)

    if prediction[0] == 1:
        return "🩺 Prediction: High Risk of Diabetes (Positive)"
    else:
        return "✅ Prediction: Low Risk of Diabetes (Negative)"


# Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Plasma Glucose (mg/dL)"),
        gr.Number(label="Diastolic Blood Pressure (mm Hg)"),
        gr.Number(label="Triceps Thickness (mm)"),
        gr.Number(label="Serum Insulin (mu U/mL)"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree"),
        gr.Number(label="Age (Years)")
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🩺 Diabetes Prediction System",
    description="""
This web application predicts the risk of diabetes using a Decision Tree Machine Learning model.

📌 Developed by: **Prachi Valecha**
🏫 College: **Panipat Institute of Engineering and Technology (PIET), Panipat**
""",
    article="""
### Instructions
- Enter all the patient's medical details.
- Click **Submit** to get the prediction.
- The prediction is generated using a trained Decision Tree Machine Learning model.

**Note:** This application is for educational purposes only and should not be considered a medical diagnosis.
"""
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
