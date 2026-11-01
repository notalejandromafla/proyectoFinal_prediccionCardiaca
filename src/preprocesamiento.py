"""Pasos 1-3 del workflow: preparar los datos.   [Integrante 1]

Salida esperada: data/train.csv y data/test.csv listos para el modelo.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from datos import cargar_datos

RAIZ = Path(__file__).resolve().parent.parent
SEMILLA = 42


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """TODO: tratar los valores faltantes (columnas 'ca' y 'thal').
    Opciones: eliminar esas filas (son pocas) o imputar con la moda.
    Documentar en el README cuál se eligió y por qué."""
    return df


def dividir(df: pd.DataFrame):
    """División 80/20 estratificada: ambos conjuntos con la misma
    proporción de enfermos y sanos."""
    X = df.drop(columns="target")
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)


if __name__ == "__main__":
    df = limpiar(cargar_datos())
    X_train, X_test, y_train, y_test = dividir(df)

    # El escalado NO se hace aquí: va dentro del Pipeline del modelo
    # para que se ajuste solo con entrenamiento (evita fuga de información).
    X_train.assign(target=y_train).to_csv(RAIZ / "data" / "train.csv", index=False)
    X_test.assign(target=y_test).to_csv(RAIZ / "data" / "test.csv", index=False)
    print(f"Entrenamiento: {len(X_train)} pacientes | Prueba: {len(X_test)} pacientes")
