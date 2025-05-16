import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("src/models/modelo_seguro.pkl")

st.title("🧠 Predicción de Costo de Seguro Médico")

# Inputs del usuario
edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
sexo = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
hijos = st.number_input("Número de hijos", min_value=0, max_value=10, value=0)
fumador = st.selectbox("¿Fumador?", ["yes", "no"])

# Botón para predecir
if st.button("Predecir costo del seguro"):
    try:
        # Codificar texto a número como en el entrenamiento
        sexo_cod = 1 if sexo == 'male' else 0
        fumador_cod = 1 if fumador == 'yes' else 0

        entrada = pd.DataFrame({
            'age': [edad],
            'bmi': [bmi],
            'children': [hijos],
            'sexo': [sexo_cod],
            'fumador': [fumador_cod]
        })

        
        entrada = entrada[modelo.feature_names_in_]

        # Predicción
        prediccion = modelo.predict(entrada)
        st.success(f"💰 Costo estimado del seguro: ${prediccion[0]:,.2f}")

    except Exception as e:
        st.error("❌ Error al procesar la predicción.")
        st.exception(e)
