import streamlit as st
import pandas as pd
import joblib

# Cargar modelo
modelo = joblib.load("src/models/modelo_seguro.pkl")  # Ajusta si es necesario

st.title("🧠 Predicción de Costo de Seguro Médico")

# Entradas del usuario
edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
sexo = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
hijos = st.number_input("Número de hijos", min_value=0, max_value=10, value=0)
fumador = st.selectbox("¿Fumador?", ["yes", "no"])

# Botón
if st.button("Predecir costo del seguro"):
    try:
        # Crear DataFrame con columnas esperadas por el modelo
        entrada = pd.DataFrame({
            'age': [edad],
            'bmi': [bmi],
            'children': [hijos],
            'sexo': [sexo],
            'fumador': [fumador]
        })

        # Reordenar columnas si hace falta
        entrada = entrada[modelo.feature_names_in_]

        # Predicción
        prediccion = modelo.predict(entrada)
        st.success(f"💰 Costo estimado del seguro: ${prediccion[0]:,.2f}")

    except Exception as e:
        st.error("❌ Ha ocurrido un error al procesar la predicción.")
        st.exception(e)
