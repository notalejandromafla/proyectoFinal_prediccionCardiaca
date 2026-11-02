import pandas as pd
import joblib
from sklearn.metrics import recall_score, precision_score

# 1. Cargar datos de prueba y separar target
test_df = pd.read_csv("data/test.csv")
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]

# 2. Cargar el modelo entrenado por Santiago
# Si el archivo se llama diferente, ajusta el nombre "modelo.pkl"
modelo = joblib.load("models/modelo.pkl")

# 3. Generar predicciones
y_pred = modelo.predict(X_test)

# 4. Análisis de Equidad por Sexo
print("\n=== Fase 3: Análisis de Equidad por Sexo ===")
mask_mujeres = X_test['sex'] == 0
print("Mujeres - Recall:", recall_score(y_test[mask_mujeres], y_pred[mask_mujeres]))
print("Mujeres - Precisión:", precision_score(y_test[mask_mujeres], y_pred[mask_mujeres]))

mask_hombres = X_test['sex'] == 1
print("\nHombres - Recall:", recall_score(y_test[mask_hombres], y_pred[mask_hombres]))
print("Hombres - Precisión:", precision_score(y_test[mask_hombres], y_pred[mask_hombres]))