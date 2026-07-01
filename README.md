## Retail Rocket · Modelo de Propensión de Compra con Machine Learning

### ¿De qué se trata?
Este es uno de mis proyectos de portfolio más desafiantes en el ámbito de Machine Learning y Analítica Predictiva. Lo armé para simular un escenario real de e-commerce utilizando datos crudos de navegación, pasando de un análisis puramente descriptivo ("qué pasó") a uno predictivo ("qué va a pasar"). 

La pregunta de negocio que guió todo el desarrollo fue: **¿Qué usuarios tienen mayor probabilidad de convertir (comprar) en los próximos 7 días basados en su historial de navegación de los últimos 30 días?**

A diferencia de los proyectos de juguete que usan datasets balanceados artificialmente, este proyecto se diseñó respetando la cruda realidad del negocio: un **desbalanceo extremo de clases (tasa de conversión natural del 0.0332%)**. Lo más difícil no fue entrenar el modelo, sino construir una cañería de datos (data pipeline) limpia, evitar fugas de información (data leakage) y evaluar el modelo desde una perspectiva financiera y de negocio, en lugar de mirar métricas de forma aislada.

---

---

### Stack técnico
- Python, pandas, numpy
- scikit-learn: `train_test_split`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `roc_auc_score`, `precision_recall_curve`
- matplotlib (visualización de curvas)

**¿Por qué no XGBoost?** Se evaluó y se descartó deliberadamente. Con ~130 casos positivos disponibles para entrenar, sumar la complejidad de tuning de un gradient boosting no iba a mover el límite matemático real del problema, que es volumen de señal, no capacidad del algoritmo. Se priorizó un Random Forest simple, metodológicamente blindado (sin leakage, sin estrategias de balanceo conflictivas) y con foco 100% en la eficiencia del umbral de decisión — más valioso para este caso que perseguir un modelo marginalmente más complejo.

### Cómo correrlo
```bash
pip install pandas numpy scikit-learn matplotlib
python 01_sanitizacion.py
python 02_carga_parquet.py
python 03_construccion_target.py
python 04_feature_engineering.py
python 05_train_eval.py
```

---

## Dataset
Datos del dataset **Retail Rocket** (Kaggle): eventos de comportamiento en un e-commerce real. El pipeline procesa millones de eventos crudos agrupados en tres tipos: `view`, `add_to_cart`, `transaction`.

| Archivo | Contenido |
| :--- | :--- |
| **events_clean.parquet** | Historial completo de eventos con timestamps estandarizados, visitorid y tipo de evento. |
| **features_and_target.parquet** | Tablón maestro de características por usuario + variable objetivo (`target`). |

- **Período cubierto:** Agosto 2015 – Septiembre 2015.
- **Ventana de Observación (Features):** 01–30 de Agosto de 2015 (30 días de comportamiento).
- **Ventana de Predicción (Target):** 31 de Agosto – 06 de Septiembre de 2015 (7 días).
- **Dimensiones finales:** 300,861 usuarios × 11 columnas.

---

## Pipeline de datos
Scripts secuenciales e independientes, cada uno responsable de una sola etapa:

1. **Sanitización y carga (Scripts 01–02):** conversión de timestamps de milisegundos a datetime; exportación a Parquet para manejar el volumen sin pérdida de registros.
2. **Construcción del target (Script 03):** ventana de predicción de 7 días sobre el corte del 31 de agosto.
3. **Ingeniería de características (Script 04):**
   - `total_pageviews`, `total_addtocart`, `total_transacciones`, `total_eventos`.
   - `dias_activos`: cantidad de jornadas con interacción.
   - `recencia_dias`: distancia en días entre el último evento del usuario y la fecha de corte. Se detectó y corrigió un bug de leakage que generaba valores negativos para eventos posteriores al corte (fix: `.clip(lower=0)`).
   - `ratio_conversion`, `ratio_intent`: relación carrito/vistas, como proxy de intención de compra.
   - `tuvo_addtocart`: flag binario de alta intención.
4. **Split y balanceo (Script 05):**
   - Train (240,688 filas) / Test (60,173 filas), con estratificación (`stratify=y`).
   - **Regla de oro:** Undersampling de la clase mayoritaria aplicado **únicamente sobre Train** (queda balanceado 50/50, 260 filas). Test queda intacto con su desbalanceo real (32 compradores sobre 60,173) para simular fielmente el comportamiento en producción.
   - Se corrigieron dos bugs de este script: (a) el modelo se estaba entrenando sobre el dataset completo sin balancear en vez del set submuestreado, y (b) convivían dos estrategias de balanceo conflictivas (`class_weight="balanced_subsample"` + undersampling manual simultáneo). Se dejó una sola estrategia activa.

---

## Resultados

**Modelo final:** Random Forest Classifier.

**ROC-AUC: 0.8139** — el modelo ordena correctamente a un comprador real por encima de un no-comprador el 81% de las veces. Es la métrica que importa acá: con 162 compradores sobre 300K+ usuarios, accuracy y precision al umbral por defecto son directamente engañosas (accuracy de 0.71 no dice nada útil cuando la clase minoritaria es el 0.05% del total).

### Importancia de features
`recencia_dias` domina el modelo con más de la mitad del peso total — es, por lejos, la señal más fuerte de intención de compra a corto plazo:

| Feature | Importancia |
| :--- | :--- |
| recencia_dias | 0.512 |
| total_eventos | 0.156 |
| total_pageviews | 0.102 |
| ratio_intent | 0.060 |
| dias_activos | 0.058 |
| total_addtocart | 0.054 |
| tuvo_addtocart | 0.040 |
| total_transacciones | 0.010 |
| ratio_conversion | 0.008 |

### Auditoría multi-umbral
Moviendo el umbral de decisión, la precisión mejora a costa de recall — el trade-off central de este modelo:

| Umbral | TP | FP | FN | Precision | Recall |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 30 | 39,441 | 2 | 0.0008 | 0.9375 |
| 0.20 | 27 | 29,417 | 5 | 0.0009 | 0.8438 |
| 0.50 (default) | 22 | 17,328 | 10 | 0.0013 | 0.6875 |
| 0.70 | 18 | 10,044 | 14 | 0.0018 | 0.5625 |
| 0.80 | 18 | 6,494 | 14 | 0.0028 | 0.5625 |
| 0.90 | 15 | 1,804 | 17 | 0.0082 | 0.4688 |

### Del conteo a la decisión de negocio
El dataset no incluye precios ni costos de contacto reales, así que en vez de inventar cifras en dólares, la forma honesta de leer esta tabla es como **ratio de equilibrio (FP necesarios por cada TP capturado)**:

| Umbral | FP por cada TP |
| :--- | ---: |
| 0.50 | ~788 |
| 0.70 | ~558 |
| 0.80 | ~361 |
| 0.90 | ~120 |

Esto traduce el modelo a una decisión operable sin inventar supuestos: **la campaña es rentable en un umbral dado si el valor de una venta capturada supera en más de ese ratio al costo de contactar a un usuario que no compra.** Con datos reales de negocio (ticket promedio, costo por email/notificación/anuncio), basta reemplazar esos dos números para saber en qué umbral operar. A 0.90, por ejemplo, el filtro es rentable en cuanto el valor de una venta sea 120 veces mayor al costo de un contacto fallido — un umbral bajo de cumplir en la mayoría de los negocios de retail online.

**Importante:** el modelo identifica probabilidad de compra, no la causa. Un recall del 47% a umbral 0.90 significa que se detecta a esa proporción de compradores que iban a convertir de todos modos — el valor de negocio está en concentrar el presupuesto de marketing donde hay más probabilidad de impacto, no en atribuir esas ventas como generadas por el modelo.

---

## Limitaciones (honestidad ante todo)
El cuello de botella de este proyecto **no es de código ni de modelado, es de datos**: apenas 162 casos positivos en todo el dataset (130 disponibles para entrenar tras el split). Con esa cantidad de señal, el techo de mejora vía feature engineering o tuning de hiperparámetros es bajo — por eso se descartó invertir tiempo en XGBoost u otras alternativas más pesadas.

## Próximos pasos
Dos caminos abiertos, sin ejecutar todavía:
1. Ampliar la ventana de observación (30+ días) para intentar capturar más compradores y mover el ROC-AUC.
2. Cerrar el proyecto en el estado actual, documentado como techo razonable dado el volumen de datos disponible — que es la decisión tomada por ahora.
