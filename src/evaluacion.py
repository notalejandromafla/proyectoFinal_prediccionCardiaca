from sklearn.metrics import recall_score, precision_score

print("\n=== Fase 3: Análisis de Equidad por Sexo ===")
mask_mujeres = X_test['sex'] == 0
print("Mujeres - Recall:", recall_score(y_test[mask_mujeres], y_pred[mask_mujeres]))
print("Mujeres - Precisión:", precision_score(y_test[mask_mujeres], y_pred[mask_mujeres]))

mask_hombres = X_test['sex'] == 1
print("\nHombres - Recall:", recall_score(y_test[mask_hombres], y_pred[mask_hombres]))
print("Hombres - Precisión:", precision_score(y_test[mask_hombres], y_pred[mask_hombres]))