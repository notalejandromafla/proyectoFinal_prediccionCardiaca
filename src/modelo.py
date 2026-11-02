import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Cargar datos de entrenamiento
train_df = pd.read_csv("data/train.csv")
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]

# Entrenar el modelo
print("Entrenando el modelo...")
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Guardar el modelo
os.makedirs("models", exist_ok=True)
joblib.dump(modelo, "models/modelo.pkl")
print("Modelo guardado en models/modelo.pkl")