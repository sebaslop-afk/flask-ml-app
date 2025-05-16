import streamlit as st
import pandas as pd
import joblib

# Cargar el modelo entrenado
modelo = joblib.load("src/models/modelo_seguro.pkl")  # ajusta la ruta si lo mueves

# Título de la app
st.title("🧠 Predicción de Costo de Seguro Médico")
st.markdown("Completa los datos del asegurado para predecir el costo estimado del seguro.")

# Entradas del usuario
edad = st.number_input("Edad", min_value=0, max_value=120, value=30)
sexo = st.selectbox("Sexo", ["male", "female"])
bmi = st.number_input("IMC (Índice de Masa Corporal)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)

# Botón para ejecutar la predicción
if st.button("Predecir costo del seguro"):
    # Codificar la entrada como fue entrenado el modelo
    entrada = pd.DataFrame({
        'age': [edad],
        'bmi': [bmi],
        'sex_male': [1 if sexo == 'male' else 0]
    })

    # Predicción
    prediccion = modelo.predict(entrada)

    # Mostrar resultado
    st.success(f"💰 Costo estimado del seguro: ${prediccion[0]:,.2f}")
