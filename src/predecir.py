"""Paso 7 del workflow: usar el modelo con datos nuevos.   [Integrante 2]

Predice para un paciente inventado (no pertenece al dataset).
"""
from pathlib import Path

import joblib
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# Paciente de ejemplo con las 13 variables (mismos códigos que el dataset)
paciente_nuevo = pd.DataFrame([{
    "age": 55, "sex": 0, "cp": 3, "trestbps": 140, "chol": 250,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
}])

if __name__ == "__main__":
    modelo = joblib.load(RAIZ / "models" / "modelo.joblib")

    prediccion = modelo.predict(paciente_nuevo)[0]
    probabilidad = modelo.predict_proba(paciente_nuevo)[0][1]

    resultado = "CON enfermedad cardíaca" if prediccion == 1 else "SIN enfermedad cardíaca"
    print(f"Predicción: paciente {resultado}")
    print(f"Probabilidad estimada de enfermedad: {probabilidad:.1%}")
    print("\nAviso: esto NO es un diagnóstico médico. Es una estimación de apoyo")
    print("que debe ser revisada por un profesional de la salud.")
