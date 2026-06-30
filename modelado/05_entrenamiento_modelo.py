import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve
)
import matplotlib.pyplot as plt

# 1. CARGAR DATOS
print("=== PASO 1: CARGA DE DATOS ===")
df = pd.read_parquet("processed/features_and_target.parquet")
print(f"-> Dimensiones: {df.shape}")
print(df["target"].value_counts())

# =========================================================================
# 2. PREPARAR X e Y
# =========================================================================
X = df.drop(columns=["visitorid", "target"])
y = df["target"]
print(f"\nFeatures usadas: {list(X.columns)}")

# 3. DIVISIÓN TRAIN / TEST
print("\n=== PASO 2: DIVISIÓN TRAIN/TEST ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"-> Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")

# 4. UNDERSAMPLING SOLO EN TRAIN
print("\n=== PASO 3: UNDERSAMPLING EN TRAIN ===")
train_temp = pd.concat([X_train, y_train], axis=1)
compradores    = train_temp[train_temp["target"] == 1]
no_compradores = train_temp[train_temp["target"] == 0]

no_compradores_sample = no_compradores.sample(n=len(compradores), random_state=42)
train_balanceado = pd.concat([compradores, no_compradores_sample])

X_train_final = train_balanceado.drop(columns=["target"])
y_train_final = train_balanceado["target"]

print(f"-> Train balanceado: {X_train_final.shape[0]} filas (50/50)")
print(f"-> Test intacto:     {X_test.shape[0]} filas")

# 5. ENTRENAMIENTO
print("\n=== PASO 4: ENTRENAMIENTO ===")
modelo = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
modelo.fit(X_train_final, y_train_final)
print("[OK] Modelo entrenado.")

# 6. EVALUACIÓN CON ANÁLISIS DE UMBRALES
print("\n=== PASO 5: EVALUACIÓN ===")
y_proba = modelo.predict_proba(X_test)[:, 1]

# Métricas con threshold default (referencia)
y_pred_default = modelo.predict(X_test)
print("\n--- UMBRAL DEFAULT (0.5) ---")
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Análisis sistemático de umbrales
print("\n--- ANÁLISIS DE UMBRALES ---")
print(f"{'Umbral':>8} | {'TP':>4} | {'FP':>6} | {'FN':>4} | {'Precision':>10} | {'Recall':>7} | {'F1':>6}")
print("-" * 65)

umbrales_a_evaluar = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

for umbral in umbrales_a_evaluar:
    y_pred_t = (y_proba >= umbral).astype(int)
    cm = confusion_matrix(y_test, y_pred_t)

    # Extraer valores del confusion matrix con protección ante matrices incompletas
    tn = cm[0, 0] if cm.shape == (2, 2) else 0
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    tp = cm[1, 1] if cm.shape == (2, 2) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"{umbral:>8.2f} | {tp:>4} | {fp:>6} | {fn:>4} | {precision:>10.4f} | {recall:>7.4f} | {f1:>6.4f}")

# 7. IMPORTANCIA DE FEATURES
print("\n=== PASO 6: IMPORTANCIA DE FEATURES ===")
importancias = pd.Series(
    modelo.feature_importances_, index=X.columns
).sort_values(ascending=False)
print(importancias.round(4))

# 8. CURVA PRECISION / RECALL
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(9, 5))
plt.plot(thresholds, precision_vals[:-1], label="Precision", color="steelblue")
plt.plot(thresholds, recall_vals[:-1],    label="Recall",    color="darkorange")
plt.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5, label="Threshold default (0.5)")
plt.xlabel("Threshold")
plt.title("Precision vs Recall por Threshold")
plt.legend()
plt.tight_layout()
plt.show()
