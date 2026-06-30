import pandas as pd
import pyarrow as pa
import numpy as np
import sklearn 
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date 

events_df = pd.read_csv("C:\\Users\\Usuario\\OneDrive\\Documentos\\curriculum yo\\Analisis de datos\\Project machine\\events.csv")
print(events_df.head())
print(events_df.info())

print(events_df[events_df['event'] == 'transaction']['visitorid'].nunique())

# Convertir el timestamp a datetime
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="ms") # convierte el timestamp a datetime
print(events_df['timestamp'].dtype)
fecha_min = events_df["timestamp"].min() # fecha mínima
fecha_max = events_df["timestamp"].max() # fecha máxima
print("Fecha mínima:", fecha_min)
print("Fecha máxima:", fecha_max)

# Cruzar los eventos con la presencia de transactionid
print(pd.crosstab(events_df['event'], events_df['transactionid'].notnull()))

# === AGREGÁ ESTA LÍNEA AL FINAL ===
events_df.to_parquet("processed/events_clean.parquet", index=False)
print("Script 01 terminado: Archivo 'events_clean.parquet' guardado con éxito.")
