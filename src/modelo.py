"""Pasos 4-5 del workflow: elegir y entrenar el modelo.   [Integrante 2]

Entrena dos modelos y los guarda en models/:
  - baseline.joblib : DummyClassifier (modelo de referencia)
  - modelo.joblib   : Regresión Logística dentro de un Pipeline
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def cargar_entrenamiento():
    df = pd.read_csv(RAIZ / "data" / "train.csv")
    return df.drop(columns="target"), df["target"]


if __name__ == "__main__":
    X_train, y_train = cargar_entrenamiento()

    # TODO 1: modelo de referencia
    #   from sklearn.dummy import DummyClassifier
    #   baseline = DummyClassifier(strategy="most_frequent")

    # TODO 2: modelo final en un Pipeline (escalado + modelo)
    #   from sklearn.pipeline import make_pipeline
    #   from sklearn.preprocessing import StandardScaler
    #   from sklearn.linear_model import LogisticRegression
    #   modelo = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # TODO 3: entrenar ambos con .fit(X_train, y_train)
    #         y guardarlos con joblib.dump(..., RAIZ / "models" / "....joblib")
    raise NotImplementedError("Completar los TODO de modelo.py")
