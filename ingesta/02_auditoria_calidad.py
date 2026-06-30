# DEFINICIÓN DE TARGET 
import pandas as pd

# 1. Cargar el archivo limpio completo generado por el Script 01
df = pd.read_parquet('processed/events_clean.parquet')

# 2. Definir fechas de corte (T es el punto de división)
T = pd.Timestamp('2015-08-31') 
obs_start = T - pd.Timedelta(days=14)  # Período de observación (Features)
pred_end = T + pd.Timedelta(days=7)    # Período de predicción (Target)

# 3. Armar las ventanas temporales analizando el volumen real
obs_window = df[(df['timestamp'] >= obs_start) & (df['timestamp'] < T)]
pred_window = df[(df['timestamp'] > T) & (df['timestamp'] <= pred_end)]

# 4. Identificar usuarios activos y compradores reales
usuarios_activos = obs_window['visitorid'].unique() 
compradores = pred_window[pred_window['event'] == 'transaction']['visitorid'].unique() 

# 5. Crear el DataFrame del Target REAL (Naturalmente desbalanceado)
target = pd.DataFrame({'visitorid': usuarios_activos}) 
target['target'] = target['visitorid'].isin(compradores).astype(int)

# 6. Guardar la población completa del negocio en disco
target.to_parquet('processed/target.parquet', index=False) 

# 7. Reporte de Sanidad para auditar en la terminal
print("=== REPORTE DE SANIDAD: SCRIPT 02 ===")
print(f"Total usuarios activos en observación: {len(target)}")
print(f"Total compradores reales encontrados: {target['target'].sum()}")
print(f"Ratio de conversión real en el negocio: {target['target'].mean():.4f}")
print("Archivo 'target.parquet' guardado con éxito en 'processed/'.")
