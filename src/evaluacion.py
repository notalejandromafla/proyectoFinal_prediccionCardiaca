"""Paso 6 del workflow: evaluar el desempeño.   [Integrante 2, con apoyo del 1 en equidad]

Calcula métricas en el conjunto de PRUEBA (datos que el modelo nunca vio).
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent


def cargar_prueba():
    df = pd.read_csv(RAIZ / "data" / "test.csv")
    return df.drop(columns="target"), df["target"]


if __name__ == "__main__":
    X_test, y_test = cargar_prueba()

    # TODO 1: cargar baseline y modelo con joblib.load y predecir sobre X_test

    # TODO 2: tabla de métricas para ambos modelos
    #   from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # TODO 3: matriz de confusión del modelo final y guardarla como imagen
    #   from sklearn.metrics import ConfusionMatrixDisplay
    #   ... plt.savefig(RAIZ / "docs" / "img" / "matriz_confusion.png")

    # TODO 4 (Integrante 1 - equidad): repetir las métricas separando por sexo
    #   mujeres = X_test["sex"] == 0 ; hombres = X_test["sex"] == 1
    raise NotImplementedError("Completar los TODO de evaluacion.py")
