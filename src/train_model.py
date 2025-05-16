from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import pandas as pd

# Cargar los datos
df = pd.read_csv('ruta/a/tu/dataset.csv')

# Codificar 'sex' con get_dummies (one-hot)
df = pd.get_dummies(df, columns=['sex'], drop_first=True)  # crea 'sex_male'

# Separar características y etiqueta
X = df.drop(columns=['charges'])
y = df['charges']

# Entrenar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Guardar modelo
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/modelo_seguro.pkl")
print("✅ Modelo guardado en 'models/modelo_seguro.pkl'")


