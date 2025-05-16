import streamlit as st
import pandas as pd
import joblib

# Cargar modelo (ajusta el nombre si tu archivo tiene otro)
modelo = joblib.load("models/modelo_entrenado.pkl")

# Título
st.title("🧠 Predicción de Costo de Seguro Médico")

st.markdown("Completa los datos del asegurado para predecir el costo.")

# Formulario de entrada
edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
sexo = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Cuando se presiona el botón
if st.button("Predecir costo del seguro"):
    entrada = pd.DataFrame({
        'age': [edad],
        'sex': [sexo],
        'bmi': [bmi]
    })

    # ⚠️ Asegúrate de que el modelo esté entrenado con las mismas columnas y codificaciones
    prediccion = modelo.predict(entrada)
    st.success(f"💰 Costo estimado del seguro: ${prediccion[0]:,.2f}")
