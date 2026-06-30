# Retail Rocket · Modelo de Propensión de Compra con Machine Learning

## ¿De qué se trata?
Este es uno de mis proyectos de portfolio más desafiantes en el ámbito de Machine Learning y Analítica Predictiva. Lo armé para simular un escenario real de e-commerce utilizando datos crudos de navegación, pasando de un análisis puramente descriptivo ("qué pasó") a uno predictivo ("qué va a pasar"). 

La pregunta de negocio que guió todo el desarrollo fue: **¿Qué usuarios tienen mayor probabilidad de convertir (comprar) en los próximos 7 días basados en su historial de navegación de los últimos 30 días?**

A diferencia de los proyectos de juguete que usan datasets balanceados artificialmente, este proyecto se diseñó respetando la cruda realidad del negocio: un **desbalanceo extremo de clases (tasa de conversión natural del 0.0332%)**. Lo más difícil no fue entrenar el modelo, sino construir una cañería de datos (data pipeline) limpia, evitar fugas de información (data leakage) y evaluar el modelo desde una perspectiva financiera y de negocio, en lugar de mirar métricas de forma aislada.

---

## Dataset
Los datos originales provienen del dataset **Retail Rocket** en Kaggle, que representa una mina de oro de eventos de comportamiento en un e-commerce real. El pipeline procesa millones de eventos crudos agrupados en tres tipos de interacciones principales: `view` (vistas), `add_to_cart` (agregar al carrito) y `transaction` (transacción/compra).

| Archivo / Tabla | Qué contiene |
| :--- | :--- |
| **events_clean.parquet** | Historial completo de eventos con timestamps estandarizados, visitorid y tipo de evento. |
| **features_and_target.parquet** | Tablón maestro consolidado de características agregadas por usuario junto con la variable objetivo (`target`). |

* **Período cubierto:** Agosto 2015 – Septiembre 2015.
* **Ventana de Observación (Features):** 01 de Agosto al 30 de Agosto de 2015 (30 días de comportamiento).
* **Ventana de Predicción (Target):** 31 de Agosto al 06 de Septiembre de 2015 (7 días para evaluar conversión).

---

## Qué hice con los datos antes de entrenar (Data Pipeline)
El procesamiento se dividió en scripts secuenciales e independientes para garantizar la modularidad y escalabilidad de la cañería:

1. **Sanitización y Carga Real (Script 01 y 02):** Conversión de timestamps de milisegundos a objetos datetime y exportación eficiente utilizando formato Parquet para manejar el volumen masivo sin pérdida de registros (asegurando el procesamiento completo de 300,861 usuarios únicos).
2. **Ingeniería de Características Avanzada (Script 04):** Construcción del tablón maestro agregando el comportamiento histórico del mes de agosto:
   * `total_pageviews`, `total_addtocart`, `total_transacciones` y `total_eventos`.
   * `dias_activos`: Cantidad de jornadas en las que interactuó el usuario.
   * `recencia`: Distancia continua en días decimales entre el último evento del usuario y la fecha de corte (31 de Agosto). Variable crítica para detectar la "frescura" del interés.
   * `ratio_conversion` y `ratio_intent`: Relación entre carritos y vistas para capturar la intención de compra.
   * `tuvo_addtocart`: Variable binaria que actúa como un flag de alta intención.
3. **División Arquitectónica y Aislamiento (Script 05):** * Se realizó un split de Train (80%) y Test (20%) preservando estrictamente el desbalanceo natural mediante **estratificación** (`stratify=y`).
   * **Regla de oro de producción:** Se aplicó *Undersampling* (submuestreo de la clase mayoritaria) **ÚNICAMENTE sobre el conjunto de entrenamiento (Train Set)** para que el algoritmo aprenda a reconocer patrones de compra. El conjunto de prueba (Test Set) permaneció intacto con sus miles de filas reales para simular fielmente el comportamiento del modelo mañana en producción.

---

## Resultados del Modelo: El Enfoque de Consultor

El modelo final implementado es un **Random Forest Classifier**. Si evaluamos el modelo con los ojos de un analista tradicional que solo mira el umbral estándar (0.50), el resultado parece catastrófico. Sin embargo, aplicando una **Auditoría Multi-Umbral Dinámica**, el modelo revela su verdadero poder para el negocio.

### Métricas Globales
* **ROC-AUC: 0.89** -> Demuestra una capacidad sobresaliente (89% de probabilidad) para ordenar y priorizar a los compradores reales por encima de los usuarios que solo entran a mirar.

### La Auditoría de Umbrales (Threshold Tuning)
Al mover la perilla de probabilidad, entendemos cómo colapsan los Falsos Positivos y empieza a respirar la precisión:

* **A Umbral 0.50 (Por defecto):** Captura 17 compradores en Test, pero genera **16,454 Falsos Positivos**. Esto destruiría el presupuesto de marketing y llenaría de spam a los usuarios.
* **A Umbral 0.80:** Los Falsos Positivos caen drásticamente a **1,789** (un bajón del 89% del ruido), atrapando a 10 compradores reales (50% Recall).
* **A Umbral 0.90 (Óptimo de Negocio):** El filtro se vuelve ultra estricto.
