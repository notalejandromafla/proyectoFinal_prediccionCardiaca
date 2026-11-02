"""Pasos 4-5 del workflow: elegir y entrenar el modelo.   [Integrante 2]

Entrena dos modelos y los guarda en models/:
  - baseline.joblib : DummyClassifier (modelo de referencia)
  - modelo.joblib   : Regresión Logística dentro de un Pipeline
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parent.parent


def cargar_entrenamiento():
    df = pd.read_csv(RAIZ / "data" / "train.csv")
    return df.drop(columns="target"), df["target"]


if __name__ == "__main__":
    X_train, y_train = cargar_entrenamiento()

    # Modelo de referencia: siempre predice la clase más frecuente (sano)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    # Modelo final: escalado + regresión logística.
    # El escalado va dentro del Pipeline para que se ajuste solo con
    # los datos de entrenamiento (evita fuga de información).
    modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    modelo.fit(X_train, y_train)

    (RAIZ / "models").mkdir(exist_ok=True)
    joblib.dump(baseline, RAIZ / "models" / "baseline.joblib")
    joblib.dump(modelo, RAIZ / "models" / "modelo.joblib")

    print(f"Modelos entrenados con {len(X_train)} pacientes y guardados en models/")
