"""Carga del dataset Heart Disease (UCI, id=45).

Base compartida del proyecto. Descarga el dataset la primera vez y lo
guarda en data/heart_disease.csv para no depender de internet después.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
RUTA_CSV = RAIZ / "data" / "heart_disease.csv"


def cargar_datos() -> pd.DataFrame:
    """Devuelve el dataset con las 13 variables y la columna 'target' (0 = sano, 1 = enfermo)."""
    if RUTA_CSV.exists():
        return pd.read_csv(RUTA_CSV)

    from ucimlrepo import fetch_ucirepo

    heart = fetch_ucirepo(id=45)
    df = heart.data.features.copy()
    # La columna original 'num' va de 0 a 4 (grado de enfermedad).
    # La convertimos a binaria: 0 = sin enfermedad, 1 = con enfermedad.
    df["target"] = (heart.data.targets["num"] > 0).astype(int)

    RUTA_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(RUTA_CSV, index=False)
    return df


if __name__ == "__main__":
    datos = cargar_datos()
    print(datos.shape)
    print(datos.head())
