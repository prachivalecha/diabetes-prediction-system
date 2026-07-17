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
    raise FileNotFoundError(
        f"Model file '{MODEL_PATH}' not found."
    )
except Exception as e:
    raise RuntimeError(
        f"Error loading model: {e}"
    )


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
    Age
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

    # Empty Check
    for value, name in zip(values, names):
        if value is None:
            return False, f"❌ {name} cannot be empty."

    # Pregnancy
    if Pregnancies < 0 or Pregnancies > 20:
        return False, "❌ Pregnancies must be between 0 and 20."

    # Glucose
    if PlasmaGlucose < 0 or PlasmaGlucose > 300:
        return False, "❌ Plasma Glucose must be between 0 and 300."

    # Blood Pressure
    if DiastolicBloodPressure < 0 or DiastolicBloodPressure > 200:
        return False, "❌ Blood Pressure must be between 0 and 200."

    # Skin Thickness
    if TricepsThickness < 0 or TricepsThickness > 100:
        return False, "❌ Triceps Thickness must be between 0 and 100."

    # Insulin
    if SerumInsulin < 0 or SerumInsulin > 1000:
        return False, "❌ Serum Insulin must be between 0 and 1000."

    # BMI
    if BMI < 5 or BMI > 80:
        return False, "❌ BMI must be between 5 and 80."

    # Diabetes Pedigree
    if DiabetesPedigree < 0 or DiabetesPedigree > 3:
        return False, "❌ Diabetes Pedigree must be between 0 and 3."

    # Age
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
            float(Age)
        ]]

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            return """
🩺 HIGH RISK OF DIABETES

Prediction: Positive

Please consult a qualified healthcare professional for proper diagnosis.
"""

        return """
✅ LOW RISK OF DIABETES

Prediction: Negative

Maintain a healthy lifestyle and regular checkups.
"""

    except Exception as e:
        return f"❌ Prediction Error\n\n{str(e)}"


# =====================================
# Gradio Interface
# =====================================

interface = gr.Interface(
    fn=predict_diabetes,

    inputs=[

        gr.Number(
            label="Pregnancies",
            minimum=0,
            maximum=20,
            precision=0
        ),

        gr.Number(
            label="Plasma Glucose (mg/dL)",
            minimum=0,
            maximum=300
        ),

        gr.Number(
            label="Diastolic Blood Pressure (mm Hg)",
            minimum=0,
            maximum=200
        ),

        gr.Number(
            label="Triceps Thickness (mm)",
            minimum=0,
            maximum=100
        ),

        gr.Number(
            label="Serum Insulin (μU/mL)",
            minimum=0,
            maximum=1000
        ),

        gr.Number(
            label="BMI",
            minimum=5,
            maximum=80
        ),

        gr.Number(
            label="Diabetes Pedigree Function",
            minimum=0,
            maximum=3
        ),

        gr.Number(
            label="Age",
            minimum=1,
            maximum=120,
            precision=0
        ),

    ],

    outputs=gr.Textbox(
        label="Prediction Result",
        lines=6
    ),

    title="🩺 Diabetes Prediction System",

    description="""
This web application predicts the likelihood of diabetes using a Decision Tree Machine Learning model.

📌 Developed by: **Prachi Valecha**

🏫 Panipat Institute of Engineering and Technology (PIET), Panipat
""",

    article="""
### Instructions

1. Enter all patient details.
2. Click **Submit**.
3. The model predicts diabetes risk.

⚠️ Educational Purpose Only.
This application should not be used as a substitute for professional medical advice.
""",

    submit_btn="Predict",
    clear_btn="Clear",

    examples=[
        [2,120,70,20,85,28.5,0.45,25],
        [6,180,90,35,250,38.4,1.2,55],
        [0,95,65,18,80,22.1,0.30,21]
    ],

    allow_flagging="never"
)

# =====================================
# Launch
# =====================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
