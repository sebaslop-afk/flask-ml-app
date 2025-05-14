from flask import Flask, render_template, request
import os
import joblib
from src.utils import db_connect

# Crear la aplicación Flask
app = Flask(__name__)

# Conexión a la base de datos (si la usas en otro lugar)
engine = db_connect()

# Cargar modelo de predicción
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "src", "models", "modelo_seguro.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print(f"⚠️ Modelo no encontrado en: {model_path}")

# Ruta principal
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para hacer la predicción
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Error: modelo no disponible", 500

    # Obtener datos del formulario
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]

    # Codificar variables categóricas
    smoker_encoded = 1 if smoker == "yes" else 0
    region_dict = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_encoded = region_dict.get(region.lower(), -1)

    # Preparar datos para la predicción
    input_data = [[age, bmi, children, smoker_encoded, region_encoded]]

    # Hacer predicción
    prediction = model.predict(input_data)[0]

    # Mostrar resultado
    return render_template("result.html", prediction=round(prediction, 2))

# Ejecutar localmente (no se usa en Render)
if __name__ == "__main__":
    app.run(debug=True)
