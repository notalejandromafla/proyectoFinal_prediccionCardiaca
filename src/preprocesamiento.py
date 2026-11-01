import pandas as pd
from sklearn.model_selection import train_test_split

def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con valores faltantes (ca y thal tienen 6 filas nulas) 
    para evitar introducir ruido por imputación artificial.
    Asegura que la variable target sea estrictamente binaria.
    """
    df_limpio = df.dropna().copy()
    df_limpio["target"] = (df_limpio["target"] > 0).astype(int)
    return df_limpio

if __name__ == "__main__":
    print("Cargando datos crudos...")
    df = pd.read_csv("data/heart_disease.csv")
    
    print("Limpiando datos...")
    df_limpio = limpiar(df)
    
    print("Dividiendo en entrenamiento y prueba (80/20 estratificado)...")
    train_df, test_df = train_test_split(
        df_limpio,
        test_size=0.20,
        random_state=42,
        stratify=df_limpio["target"]
    )
    
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("¡Listo! Archivos train.csv y test.csv guardados en la carpeta data/")