import os
from flask import Flask, render_template, request
import joblib
from src.utils import db_connect

#link https://flask-ml-app-krk4.onrender.com

app = Flask(__name__)


engine = db_connect()


try:
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "src", "models", "modelo_seguro.pkl")
    model = joblib.load(model_path)
except FileNotFoundError:
    model = None
    print("⚠️ Modelo no encontrado. Asegúrate de que 'modelo_seguro.pkl' exista.")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return "Modelo no cargado."

    
    age = int(request.form["age"])
    sex = request.form["sex"]
    bmi = float(request.form["bmi"])
    children = int(request.form["children"])
    smoker = request.form["smoker"]
    region = request.form["region"]

    
    smoker_encoded = 1 if smoker == "yes" else 0
    region_dict = {"northeast": 0, "northwest": 1, "southeast": 2, "southwest": 3}
    region_encoded = region_dict.get(region.lower(), -1)

    input_data = [[age, bmi, children, smoker_encoded, region_encoded]]

    prediction = model.predict(input_data)[0]
    return render_template("result.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
