import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("src/models/modelo_seguro.pkl")

# Título
st.title("🧠 Predicción de Costo de Seguro Médico")
st.markdown("Completa los datos del asegurado para predecir el costo estimado del seguro.")

# Inputs
edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
sexo = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
hijos = st.number_input("Número de hijos", min_value=0, max_value=10, value=0)
fumador = st.selectbox("¿Fumador?", ["yes", "no"])

# Botón
if st.button("Predecir costo del seguro"):
    try:
        # Codificar variables categóricas
        entrada = pd.DataFrame({
            'age': [edad],
            'bmi': [bmi],
            'children': [hijos],
            'sex_male': [1 if sexo == 'male' else 0],
            'smoker_yes': [1 if fumador == 'yes' else 0]
        })

        # Asegurar el orden correcto de columnas
        entrada = entrada[modelo.feature_names_in_]

        # Predicción
        prediccion = modelo.predict(entrada)
        st.success(f"💰 Costo estimado del seguro: ${prediccion[0]:,.2f}")

    except Exception as e:
        st.error("❌ Ha ocurrido un error al procesar la predicción.")
        st.exception(e)
