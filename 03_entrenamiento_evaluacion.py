import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_curve
from sklearn.utils import resample

# 1. CARGA DE DATOS
print("Cargando matriz de features y target...")
df_completo = pd.read_parquet("processed/features_and_target.parquet")

# Separar variables de entrenamiento (X) y objetivo (y)
# Eliminamos 'visitorid' porque es un identificador, no una feature predictiva
X = df_completo.drop(columns=["visitorid", "target"])
y = df_completo["target"]

# 2. TRAIN/TEST SPLIT (Mantenemos la distribución desbalanceada real en Test para auditar el negocio)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. UNDERSAMPLING SOLO EN TRAIN (Blindaje estricto contra fugas de información)
print(f"\nDistribución original en Train: {np.bincount(y_train)}")

X_train_df = pd.DataFrame(X_train)
X_train_df['target'] = y_train

df_majority = X_train_df[X_train_df['target'] == 0]
df_minority = X_train_df[X_train_df['target'] == 1]

# Submuestreamos la clase mayoritaria (0) para igualar a la minoritaria (1)
df_majority_downsampled = resample(df_majority, 
                                   replace=False,    
                                   n_samples=len(df_minority), 
                                   random_state=42)

df_balanced = pd.concat([df_majority_downsampled, df_minority])
X_train_final = df_balanced.drop(columns=['target'])
y_train_final = df_balanced['target']

print(f"Distribución balanceada en Train: {np.bincount(y_train_final)}")
print(f"Distribución real intacta en Test: {np.bincount(y_test)}")

# 4. ENTRENAMIENTO
print("\n=== PASO 4: ENTRENAMIENTO ===")
modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
modelo.fit(X_train_final, y_train_final)
print("[OK] Modelo Random Forest entrenado exitosamente.")

# 5. EVALUACIÓN CON ANÁLISIS DE UMBRALES
print("\n=== PASO 5: EVALUACIÓN ===")
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas con threshold default (0.5) para tener la referencia básica
y_pred_default = modelo.predict(X_test)
print("\n--- UMBRAL DEFAULT (0.5) ---")
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

# Análisis sistemático de umbrales para optimización del negocio
print("\n--- ANÁLISIS DE UMBRALES EN COMPORTAMIENTO ---")
print(f"{'Umbral':>8} | {'TP':>4} | {'FP':>6} | {'FN':>4} | {'Precision':>10} | {'Recall':>7} | {'F1':>6}")
print("-" * 65)

umbrales_a_evaluar = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

for umbral in umbrales_a_evaluar:
    y_pred_t = (y_proba >= umbral).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)

    # Extracción segura de la matriz de confusión
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"{umbral:>8.2f} | {tp:>4} | {fp:>6} | {fn:>4} | {precision:>10.4f} | {recall:>7.4f} | {f1:>6.4f}")

# 6. IMPORTANCIA DE FEATURES
print("\n=== PASO 6: IMPORTANCIA DE VARIABLES (FEATURE IMPORTANCE) ===")
importancias = pd.Series(
    modelo.feature_importances_, index=X.columns
).sort_values(ascending=False)
print(importancias.round(4))

# 7. CURVA PRECISION / RECALL
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(9, 5))
plt.plot(thresholds, precision_vals[:-1], label="Precision", color="steelblue", lw=2)
plt.plot(thresholds, recall_vals[:-1],    label="Recall",    color="darkorange", lw=2)
plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold default (0.5)")
plt.xlabel("Umbral de Decisión (Threshold)")
plt.ylabel("Score")
plt.title("Curva Precision vs Recall por Umbral de Predicción")
plt.legend()
plt.tight_layout()
plt.show()
