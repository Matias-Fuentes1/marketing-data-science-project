import pandas as pd
import numpy as np

print("Cargando eventos limpios...")
events = pd.read_parquet("processed/events_clean.parquet")

if events["timestamp"].dtype in [np.int64, np.float64]:
    events["timestamp"] = pd.to_datetime(events["timestamp"], unit='ms')
else:
    events["timestamp"] = pd.to_datetime(events["timestamp"])

inicio_features = "2015-08-01 00:00:00"
fin_features    = "2015-08-30 23:59:59"
inicio_target   = "2015-08-31 00:00:00"
fin_target      = "2015-09-06 23:59:59"
fecha_corte     = pd.Timestamp(fin_features)

print(f"-> Ventana de features: {inicio_features} a {fin_features}")
print(f"-> Ventana de target:   {inicio_target} a {fin_target}")

df_features_period = events[
    (events["timestamp"] >= inicio_features) &
    (events["timestamp"] <= fin_features)
].copy()

df_target_period = events[
    (events["timestamp"] >= inicio_target) &
    (events["timestamp"] <= fin_target)
].copy()

# --- MATRIZ BASE (un solo escaneo) ---
matriz_feature = pd.crosstab(
    index=df_features_period["visitorid"],
    columns=df_features_period["event"]
).reset_index()

matriz_feature.rename(columns={
    "view": "total_pageviews",
    "addtocart": "total_addtocart",
    "transaction": "total_transacciones"
}, inplace=True)

for col in ["total_pageviews", "total_addtocart", "total_transacciones"]:
    if col not in matriz_feature.columns:
        matriz_feature[col] = 0

matriz_feature["total_eventos"] = (
    matriz_feature["total_pageviews"] +
    matriz_feature["total_addtocart"] +
    matriz_feature["total_transacciones"]
)

# --- DÍAS ACTIVOS ---
df_features_period["fecha"] = df_features_period["timestamp"].dt.date
df_dias = df_features_period.groupby("visitorid")["fecha"].nunique().reset_index()
df_dias.rename(columns={"fecha": "dias_activos"}, inplace=True)
matriz_feature = matriz_feature.merge(df_dias, on="visitorid", how="left")
matriz_feature["dias_activos"] = matriz_feature["dias_activos"].fillna(1)

# --- RECENCIA (un solo groupby sobre df_features_period) ---
df_recencia = df_features_period.groupby("visitorid")["timestamp"].max().reset_index()
df_recencia["recencia_dias"] = (
    (fecha_corte - df_recencia["timestamp"]).dt.total_seconds() / 86400
).clip(lower=0)
df_recencia = df_recencia[["visitorid", "recencia_dias"]]
matriz_feature = matriz_feature.merge(df_recencia, on="visitorid", how="left")
matriz_feature["recencia_dias"] = matriz_feature["recencia_dias"].fillna(30.0)

# --- FEATURES DERIVADAS DIRECTAMENTE DE LO QUE YA TENEMOS ---
# ratio_conversion: transacciones pasadas / pageviews
matriz_feature["ratio_conversion"] = (
    matriz_feature["total_transacciones"] / matriz_feature["total_pageviews"]
).replace([np.inf, -np.inf], np.nan).fillna(0)

# ratio_intent: add-to-cart / pageviews (calculado sobre columnas ya existentes)
matriz_feature["ratio_intent"] = (
    matriz_feature["total_addtocart"] / (matriz_feature["total_pageviews"] + 1)
)

# tuvo_addtocart: binaria derivada directamente de total_addtocart
matriz_feature["tuvo_addtocart"] = (matriz_feature["total_addtocart"] > 0).astype(int)

# --- TARGET ---
compradores_futuros = df_target_period[
    df_target_period["event"] == "transaction"
]["visitorid"].unique()

matriz_feature["target"] = matriz_feature["visitorid"].isin(compradores_futuros).astype(int)

print("\n=== MATRIZ COMPLETA ===")
print("Dimensiones:", matriz_feature.shape)
print("Distribución target:\n", matriz_feature["target"].value_counts())
print(f"Tasa de conversión: {matriz_feature['target'].mean() * 100:.4f}%")

matriz_feature.to_parquet("processed/features_and_target.parquet")
print("\n[OK] Guardado en 'processed/features_and_target.parquet'")
