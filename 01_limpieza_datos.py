import os
import pandas as pd

# Leer el archivo CSV
events_df = pd.read_csv("data/events.csv")
print(events_df.head())
print(events_df.info()) 

# Contar visitantes únicos con transacciones
print(events_df[events_df['event'] == 'transaction']['visitorid'].nunique())

# Convertir el timestamp a datetime
events_df["timestamp"] = pd.to_datetime(events_df["timestamp"], unit="ms") 
print(events_df['timestamp'].dtype)
fecha_min = events_df["timestamp"].min() 
fecha_max = events_df["timestamp"].max() 
print("Fecha mínima:", fecha_min)
print("Fecha máxima:", fecha_max)

# Cruzar los eventos con la presencia de transactionid
print(pd.crosstab(events_df['event'], events_df['transactionid'].notnull()))

# Exportar a formato Parquet
os.makedirs("processed", exist_ok=True)
events_df.to_parquet("processed/events_clean.parquet", index=False)
print("Script 01 terminado: Archivo 'events_clean.parquet' guardado con éxito.")

