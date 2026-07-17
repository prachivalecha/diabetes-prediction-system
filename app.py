import os
import joblib
import gradio as gr

# =====================================
# Load Model Safely
# =====================================

MODEL_PATH = "diabetes_prediction_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


# =====================================
# Validation Function
# =====================================

def validate_inputs(
    Pregnancies,
    PlasmaGlucose,
    DiastolicBloodPressure,
    TricepsThickness,
    SerumInsulin,
    BMI,
    DiabetesPedigree,
    Age,
):

    values = [
        Pregnancies,
        PlasmaGlucose,
        DiastolicBloodPressure,
        TricepsThickness,
        SerumInsulin,
        BMI,
        DiabetesPedigree,
        Age,
    ]

    names = [
        "Pregnancies",
        "Plasma Glucose",
        "Blood Pressure",
        "Triceps Thickness",
        "Serum Insulin",
        "BMI",
        "Diabetes Pedigree",
        "Age",
    ]

    for value, name in zip(values, names):
        if value is None:
            return False, f"❌ {name} cannot be empty."

    if Pregnancies < 0 or Pregnancies > 20:
        return False, "❌ Pregnancies must be between 0 and 20."

    if PlasmaGlucose < 0 or PlasmaGlucose > 300:
        return False, "❌ Plasma Glucose must be between 0 and 300."

    if DiastolicBloodPressure < 0 or DiastolicBloodPressure > 200:
        return False, "❌ Blood Pressure must be between 0 and 200."

    if TricepsThickness < 0 or TricepsThickness > 100:
        return False, "❌ Triceps Thickness must be between 0 and 100."

    if SerumInsulin < 0 or SerumInsulin > 1000:
        return False, "❌ Serum Insulin must be between 0 and 1000."

    if BMI < 5 or BMI > 80:
        return False, "❌ BMI must be between 5 and 80."

    if DiabetesPedigree < 0 or DiabetesPedigree > 3:
        return False, "❌ Diabetes Pedigree must be between 0 and 3."

    if Age < 1 or Age > 120:
        return False, "❌ Age must be between 1 and 120."

    return True, ""


# =====================================
# Prediction Function
# =====================================

def predict_diabetes(
    Pregnancies,
    PlasmaGlucose,
    DiastolicBloodPressure,
    TricepsThickness,
    SerumInsulin,
    BMI,
    DiabetesPedigree,
    Age,
):

    valid, message = validate_inputs(
        Pregnancies,
        PlasmaGlucose,
        DiastolicBloodPressure,
        TricepsThickness,
        SerumInsulin,
        BMI,
        DiabetesPedigree,
        Age,
    )

    if not valid:
        return message

    try:
        input_data = [[
            float(Pregnancies),
            float(PlasmaGlucose),
            float(DiastolicBloodPressure),
            float(TricepsThickness),
            float(SerumInsulin),
            float(BMI),
            float(DiabetesPedigree),
            float(Age),
        ]]

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            return (
                "🩺 HIGH RISK OF DIABETES\n\n"
                "Prediction: Positive\n\n"
                "Please consult a qualified healthcare professional."
            )
        else:
            return (
                "✅ LOW RISK OF DIABETES\n\n"
                "Prediction: Negative\n\n"
                "Maintain a healthy lifestyle and regular checkups."
            )

    except Exception as e:
        return f"❌ Prediction Error:\n\n{e}"


# =====================================
# Gradio Interface
# =====================================

interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[
        gr.Number(label="Pregnancies"),
        gr.Number(label="Plasma Glucose (mg/dL)"),
        gr.Number(label="Diastolic Blood Pressure (mm Hg)"),
        gr.Number(label="Triceps Thickness (mm)"),
        gr.Number(label="Serum Insulin (μU/mL)"),
        gr.Number(label="BMI"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age (Years)")
    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=6
    ),

    title="🩺 Diabetes Prediction System",

    description="""
This web application predicts the risk of diabetes using a Decision Tree Machine Learning model.

📌 Developed by: Prachi Valecha

🏫 Panipat Institute of Engineering and Technology (PIET), Panipat
""",

    article="""
### Instructions

• Enter all patient details.

• Click Submit.

• The prediction is generated using a trained Decision Tree model.

⚠️ This application is for educational purposes only and should not be used as a medical diagnosis.
""",

    examples=[
        [2,120,70,20,85,28.5,0.45,25],
        [6,180,90,35,250,38.4,1.20,55],
        [0,95,65,18,80,22.1,0.30,21]
    ]
)

# =====================================
# Launch
# =====================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port
    )
