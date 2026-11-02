"""Paso 6 del workflow: evaluar el desempeño.   [Integrante 2, con apoyo del 1 en equidad]

Calcula métricas en el conjunto de PRUEBA (datos que el modelo nunca vio).
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RAIZ = Path(__file__).resolve().parent.parent


def cargar_prueba():
    df = pd.read_csv(RAIZ / "data" / "test.csv")
    return df.drop(columns="target"), df["target"]


def calcular_metricas(y_real, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_real, y_pred),
        # zero_division=0: el baseline nunca predice "enfermo", así que su
        # precision no se puede calcular; se reporta como 0 en vez de error.
        "Precision": precision_score(y_real, y_pred, zero_division=0),
        "Recall": recall_score(y_real, y_pred),
        "F1-Score": f1_score(y_real, y_pred, zero_division=0),
    }


if __name__ == "__main__":
    X_test, y_test = cargar_prueba()

    baseline = joblib.load(RAIZ / "models" / "baseline.joblib")
    modelo = joblib.load(RAIZ / "models" / "modelo.joblib")

    pred_baseline = baseline.predict(X_test)
    pred_modelo = modelo.predict(X_test)

    # Tabla de métricas de ambos modelos
    tabla = pd.DataFrame({
        "Referencia (DummyClassifier)": calcular_metricas(y_test, pred_baseline),
        "Regresión Logística": calcular_metricas(y_test, pred_modelo),
    }).T.round(3)
    print("=== Métricas en el conjunto de prueba ===")
    print(tabla.to_string())
    tabla.to_csv(RAIZ / "docs" / "metricas.csv")

    # Matriz de confusión del modelo final
    vn, fp, fn, vp = confusion_matrix(y_test, pred_modelo).ravel()
    print(f"\nVP={vp}  FP={fp}  VN={vn}  FN={fn}")

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred_modelo, display_labels=["Sano (0)", "Enfermo (1)"], cmap="Blues"
    )
    disp.ax_.set_title("Matriz de confusión - Regresión Logística")
    disp.ax_.set_xlabel("Predicción del modelo")
    disp.ax_.set_ylabel("Valor real")
    (RAIZ / "docs" / "img").mkdir(parents=True, exist_ok=True)
    plt.savefig(RAIZ / "docs" / "img" / "matriz_confusion.png", dpi=150, bbox_inches="tight")
    print("Matriz guardada en docs/img/matriz_confusion.png")

    # TODO 4 (Integrante 1 - equidad): repetir las métricas separando por sexo
    #   mujeres = X_test["sex"] == 0 ; hombres = X_test["sex"] == 1

from sklearn.metrics import recall_score

print("\n=== Fase 3: Análisis de Equidad por Sexo ===")
# Generamos las predicciones explícitamente para este cálculo
y_pred = modelo.predict(X_test)

mask_mujeres = X_test['sex'] == 0
recall_mujeres = recall_score(y_test[mask_mujeres], y_pred[mask_mujeres])
print("Mujeres - Recall:", recall_mujeres)

mask_hombres = X_test['sex'] == 1
recall_hombres = recall_score(y_test[mask_hombres], y_pred[mask_hombres])
print("\nHombres - Recall:", recall_hombres)