from flask import Flask, render_template, request

import joblib
from src.utils import db_connect


# Conexión a la base de datos (si es necesario para otras operaciones)
engine = db_connect()

# Cargar el modelo entrenado
model = joblib.load("models/modelo_seguro.pkl")

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta principal
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para hacer la predicción
@app.route("/predict", methods=["POST"])
def predict():
    # Obtener los datos del formulario
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]

    # Codificar las variables categóricas
    sex_encoded = 1 if sex == "male" else 0
    smoker_encoded = 1 if smoker == "yes" else 0
    region_dict = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_encoded = region_dict.get(region.lower(), -1)

    # Preparar los datos para la predicción
    input_data = [[age, bmi, children, smoker_encoded, region_encoded]]  # Eliminar sex_encoded

    # Hacer la predicción
    prediction = model.predict(input_data)[0]

    # Retornar el resultado redondeado a 2 decimales
    return render_template("result.html", prediction=round(prediction, 2))

# Ejecutar la aplicación Flask
if __name__ == "__main__":
    app.run(debug=True)
