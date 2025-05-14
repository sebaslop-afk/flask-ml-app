from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd

# Cargar los datos
df = pd.read_csv('ruta/a/tu/dataset.csv')

# Suponiendo que la columna 'charges' es la que quieres predecir
X = df.drop(columns=['charges'])  # Características
y = df['charges']  # Etiquetas

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Crear carpeta 'models' si no existe
os.makedirs("models", exist_ok=True)

# Guardar el modelo
joblib.dump(model, "models/modelo_seguro.pkl")
print("✅ Modelo guardado en 'models/modelo_seguro.pkl'")

