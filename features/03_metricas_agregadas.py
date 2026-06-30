import pandas as pd
import numpy as np

# 1. Cargar datasets
target_balanced = pd.read_parquet("processed/target_balanced.parquet")
events = pd.read_parquet("processed/events_clean.parquet")

# 2. Filtro de usuarios
usuarios_balanceados = target_balanced["visitorid"]
events_filtrado = events[events["visitorid"].isin(usuarios_balanceados)].copy()

# 3. Filtro de tiempo: Ventana de observación estricta (Parseo único de fecha)
# Si viene como entero/milisegundos desde el origen, lo tratamos correctamente de entrada
if events_filtrado["timestamp"].dtype in [np.int64, np.float64]:
    events_filtrado["timestamp"] = pd.to_datetime(events_filtrado["timestamp"], unit='ms')
else:
    events_filtrado["timestamp"] = pd.to_datetime(events_filtrado["timestamp"])

events_filtrado = events_filtrado[
    (events_filtrado["timestamp"] >= "2015-08-17")
    & (events_filtrado["timestamp"] <= "2015-08-31")
]

# 4. Calcular la matriz de eventos (con crosstab)
matriz_feature = pd.crosstab(
    index=events_filtrado["visitorid"],
    columns=events_filtrado["event"]
)

# 5. Convertir el resultado en un DataFrame limpio
matriz_feature.reset_index(inplace=True)

# 6. Renombrar columnas respetando el formato real del dataset (add_to_cart)
matriz_feature.rename(columns={
    "view": "total_pageviews",
    "add_to_cart": "total_addtocart",
    "transaction": "total_transacciones"
}, inplace=True)

# Forzar la existencia de columnas clave por si algún usuario no interactuó en algún evento específico
for columna in ["total_pageviews", "total_addtocart", "total_transacciones"]:
    if columna not in matriz_feature.columns:
        matriz_feature[columna] = 0

# Calcular el total de eventos acumulados de forma explícita
matriz_feature["total_eventos"] = (
    matriz_feature["total_pageviews"] + 
    matriz_feature["total_addtocart"] + 
    matriz_feature["total_transacciones"]
)

# 7. Calcular los días activos reales usando la columna datetime ya parseada
events_filtrado['fecha'] = events_filtrado['timestamp'].dt.date
df_dias_activos = events_filtrado.groupby("visitorid")["fecha"].nunique().reset_index()
df_dias_activos.rename(columns={"fecha": "dias_activos"}, inplace=True)

# Fusionar la matriz de features con los días activos
matriz_feature = matriz_feature.merge(df_dias_activos, on="visitorid", how="left")
matriz_feature["dias_activos"] = matriz_feature["dias_activos"].fillna(1)

# 8. Calcular el ratio de conversión matemático seguro
# 8. Calcular el ratio de conversión matemático seguro
matriz_feature["ratio_conversion"] = matriz_feature["total_transacciones"] / matriz_feature["total_pageviews"]
matriz_feature["ratio_conversion"] = matriz_feature["ratio_conversion"].replace([np.inf, -np.inf], np.nan).fillna(0)

# FUSIONAR CON EL TARGET Y GUARDAR EL DATASET FINAL 

# 9. Traer de vuelta la columna 'target' uniendo por 'visitorid'
# Usamos 'inner' join para conservar exactamente la proporción del submuestreo balanceado
dataset_final = target_balanced.merge(matriz_feature, on="visitorid", how="inner")

# Verificaciones de control en la terminal
print("\n=== AUDITORÍA DEL DATASET FINAL ===")
print("Dimensiones del dataset final para el modelo:", dataset_final.shape)
print("Distribución real de clases en el dataset final:\n", dataset_final["target"].value_counts())
print(dataset_final.head())

# 10. Guardar el archivo definitivo para que los scripts 04 y 05 operen sobre él
# NOTA: Asegurate de que este nombre coincida con el que carga tu Script 04
dataset_final.to_parquet("processed/features_train.parquet", index=False)
print("\n¡Pipeline corregido! Dataset guardado exitosamente en 'processed/features_train.parquet'.")
