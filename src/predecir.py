"""Paso 7 del workflow: usar el modelo con datos nuevos.   [Integrante 2]

Predice para un paciente inventado (no pertenece al dataset).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent

# TODO: completar con valores de ejemplo para las 13 variables
paciente_nuevo = pd.DataFrame([{
    "age": 55, "sex": 0, "cp": 3, "trestbps": 140, "chol": 250,
    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 1.0, "slope": 2, "ca": 0, "thal": 3,
}])

if __name__ == "__main__":
    # TODO: cargar models/modelo.joblib y mostrar
    #   - la predicción (0 = sano, 1 = enfermo) con .predict
    #   - la probabilidad con .predict_proba
    # Recordar en la salida: "Esto NO es un diagnóstico médico".
    raise NotImplementedError("Completar los TODO de predecir.py")
