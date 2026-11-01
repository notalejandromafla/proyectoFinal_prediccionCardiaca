# proyectoFinal_prediccionCardiaca

> Mini‑proyecto de Machine Learning: predicción de enfermedad cardíaca a partir de datos clínicos, con evaluación de métricas y análisis ético.

**Integrantes**

| Nombre | Rama | Rol |
|---|---|---|
| _Pablo Alejandro Mafla Ayala_ | `dev_pmafla58` | Datos y ética |
| _Nombre integrante 2_ | `dev_xxxx` | Modelo y evaluación |

**Documentos del proyecto**

- Informe completo (PDF): [`docs/informe.pdf`](docs/informe.pdf)
- Notebook de datos: [`notebooks/01_datos.ipynb`](notebooks/01_datos.ipynb)
- Notebook del modelo: [`notebooks/02_modelo.ipynb`](notebooks/02_modelo.ipynb)

---

## 1. Problema, datos y solución

### 1.1 ¿Qué problema resolvemos?
Las enfermedades cardiovasculares constituyen una de las principales causas de mortalidad a nivel global, y su detección temprana resulta crítica para mejorar el pronóstico clínico de los pacientes. Este proyecto implementa un modelo supervisado de clasificación binaria basado en Regresión Logística para predecir la presencia o ausencia de enfermedad cardíaca (`target`: 1 = presencia, 0 = ausencia) a partir de 13 variables clínicas y demográficas del dataset Heart Disease (UCI). El sistema está diseñado estrictamente como una herramienta de soporte a la decisión clínica (triaje preliminar) y en ningún caso reemplaza el diagnóstico ni el criterio de un profesional médico.

### 1.2 ¿Con qué datos?
_TODO (Integrante 1): nombre del dataset, fuente, número de pacientes y variables, variable objetivo. Tabla con las variables:_

| Variable | Descripción | Tipo |
|---|---|---|
| age | Edad en años | Numérica |
| sex | Sexo (1 = hombre, 0 = mujer) | Categórica |
| ... | ... | ... |
| target | 1 = tiene enfermedad cardíaca, 0 = no | Objetivo |

### 1.3 ¿Cómo funciona la solución?
_TODO (Integrante 2): describir el flujo: carga → preparación → división entrenamiento/prueba → entrenamiento → evaluación → predicción con un paciente nuevo._

### 1.4 ¿Cómo demostramos que los resultados son válidos?
_TODO (Integrante 2): conjunto de prueba que el modelo nunca vio, comparación contra un modelo de referencia, criterios de aceptación (ver sección 4)._

### 1.5 Cómo ejecutar el proyecto

El proyecto se desarrolló y ejecutó en **Amazon SageMaker** (instancia de notebook `ml.t3.medium`, kernel `conda_python3`) dentro de AWS Academy Learner Lab. También funciona en cualquier computador con Python 3.10 o superior.

En la terminal de JupyterLab:

```bash
cd ~/SageMaker/proyectoFinal_prediccionCardiaca
source activate python3          # solo en SageMaker: usa el mismo Python que los notebooks
pip install -r requirements.txt
python src/datos.py              # descarga el dataset de UCI (solo la primera vez)
python src/preprocesamiento.py   # prepara y divide los datos
python src/modelo.py             # entrena el modelo de referencia y el modelo final
python src/evaluacion.py         # calcula métricas y matriz de confusión
python src/predecir.py           # predice para un paciente nuevo
```

---

## 2. Alcance del sistema

### ✅ Qué hará el sistema
* Descargar, limpiar y particionar de forma estratificada (80% entrenamiento / 20% prueba) el dataset clínico de enfermedades cardíacas de la UCI.
* Entrenar un pipeline reproducible que integra estandarización de características (`StandardScaler`) y un modelo de Regresión Logística interpretable.
* Evaluar cuantitativamente el desempeño del clasificador frente a un modelo base (`DummyClassifier`) utilizando Accuracy, Precision, Recall, F1-Score y Matriz de Confusión.
* Auditar la equidad del modelo desglosando métricas de sensibilidad (Recall) y precisión por sexo biológico.
* Generar predicciones individuales con probabilidades asociadas para pacientes nuevos mediante consola.
  
### ❌ Qué NO hará el sistema
* **No emitirá diagnósticos médicos definitivos** ni prescribirá tratamientos clínicos.
* **No procesará historias clínicas con datos personales identificables (PII)** ni se conectará en tiempo real a sistemas hospitalarios (HIS/EHR).
* **No operará como interfaz gráfica web ni API en producción continua**, limitándose a ejecución por scripts y notebooks en el entorno controlado de Amazon SageMaker.
  
### 2.1 Diagrama C4 de contexto

```mermaid
C4Context
    title Diagrama de Contexto (C4) — Sistema de Predicción de Enfermedad Cardíaca

    Person(medico, "Personal Médico / Investigador", "Ejecuta el análisis, evalúa métricas e interpreta la predicción clínica como apoyo diagnóstico.")
    
    System(sistema_ml, "Sistema ML de Predicción Cardíaca", "Preprocesa datos clínicos, entrena el modelo de Regresión Logística, evalúa equidad y estima la probabilidad de enfermedad cardíaca (AWS SageMaker).")
    
    System_Ext(uci_repo, "UCI Machine Learning Repository", "Fuente externa de datos públicos que provee el dataset Heart Disease (Cleveland, 303 registros).")

    Rel(medico, sistema_ml, "Ejecuta scripts/notebooks, ingresa variables de paciente y consulta métricas", "CLI / JupyterLab")
    Rel(sistema_ml, uci_repo, "Descarga el dataset clínico crudo (heart_disease.csv)", "HTTPS / ucimlrepo")
```mermaid
flowchart LR
    U["👤 Profesional de salud / analista<br/>[Persona]<br/>Ingresa datos clínicos de un paciente<br/>y consulta la predicción"]
    S["🫀 Sistema de predicción cardíaca<br/>[Sistema de software]<br/>Prepara los datos, entrena el modelo<br/>y estima el riesgo de enfermedad"]
    D["🗄️ UCI Machine Learning Repository<br/>[Sistema externo]<br/>Fuente del dataset Heart Disease"]
    A["☁️ Amazon SageMaker (AWS Academy)<br/>[Sistema externo]<br/>Entorno donde se ejecutan los notebooks<br/>y se entrena el modelo"]
    G["🐙 GitHub<br/>[Sistema externo]<br/>Código, documentación y modelo versionados"]

    U -- "Envía datos del paciente /<br/>recibe la predicción" --> S
    S -- "Descarga el dataset<br/>(ucimlrepo / CSV)" --> D
    S -- "Se ejecuta en" --> A
    S -- "Se versiona en" --> G

    classDef persona fill:#08427b,color:#fff,stroke:#052e56
    classDef sistema fill:#1168bd,color:#fff,stroke:#0b4884
    classDef externo fill:#999999,color:#fff,stroke:#6b6b6b
    class U persona
    class S sistema
    class D,A,G externo
```

---

## 3. Decisiones de arquitectura (ADR)

_Formato de cada ADR: Contexto → Decisión → Justificación → Consecuencias._

### ADR‑01: Enfoque de implementación
- **Contexto:** _TODO (Integrante 2)_
- **Decisión:** _ej. Python + scikit‑learn en Amazon SageMaker (AWS Academy), con scripts en `src/` y notebooks de JupyterLab para la exploración._
- **Justificación:** _TODO_
- **Consecuencias:** _TODO_

### ADR‑02: Selección del modelo
- **Contexto:** _TODO_
- **Decisión:** _ej. Regresión Logística._
- **Justificación:** _ej. problema de clasificación binaria, dataset pequeño, modelo interpretable (importante en salud)._
- **Consecuencias:** _ej. puede no capturar relaciones no lineales._

### ADR‑03: Almacenamiento y despliegue
- **Contexto:** _TODO_
- **Decisión:** _ej. dataset descargado desde UCI y guardado en `data/`; modelo guardado con `joblib` en `models/`; ejecución en una instancia de notebook de SageMaker (`ml.t3.medium`); GitHub como almacenamiento principal y respaldo._
- **Justificación:** _ej. el dataset es pequeño y no necesita una base de datos; `ml.t3.medium` basta para entrenar en segundos y cuida el presupuesto del laboratorio; GitHub protege el trabajo si la sesión de AWS se cierra._
- **Consecuencias:** _ej. el modelo no queda expuesto como servicio web (no hay endpoint de SageMaker); para usarlo hay que ejecutar `src/predecir.py`._

---

## 4. Calidad y evaluación del modelo

### 4.1 Estrategia de separación de datos
_TODO (Integrante 2): ej. 80 % entrenamiento / 20 % prueba, estratificado por la variable objetivo, `random_state=42`. El escalado se ajusta solo con entrenamiento para evitar fuga de información._

### 4.2 Métricas

| Modelo | Accuracy | Precision | Recall | F1‑Score |
|---|---|---|---|---|
| Referencia (DummyClassifier) | _ | _ | _ | _ |
| Regresión Logística | _ | _ | _ | _ |

_TODO (Integrante 2): interpretar cada métrica en una frase. Explicar por qué el Recall es la métrica prioritaria (un falso negativo = enfermo clasificado como sano)._

### 4.3 Matriz de confusión
_TODO (Integrante 2): insertar imagen `docs/img/matriz_confusion.png` y explicar VP, FP, VN y FN._

### 4.4 Comparación con el modelo de referencia
_TODO (Integrante 2)_

### 4.5 Criterios de aceptación
- [ ] _ej. Recall ≥ 0.80 en el conjunto de prueba._
- [ ] _ej. F1‑Score ≥ 0.75._
- [ ] _ej. Supera al modelo de referencia en todas las métricas._
- [ ] _ej. La diferencia de Recall entre hombres y mujeres es menor a 0.10._
- [ ] _ej. El proyecto se ejecuta de principio a fin con los comandos de la sección 1.5._

---

## 5. Riesgos técnicos

| # | Riesgo | Impacto | Mitigación |
|---|---|---|---|
| R1 | Datos insuficientes (~300 pacientes) | _TODO_ | _TODO_ |
| R2 | Fuga de información entre entrenamiento y prueba | _TODO_ | _TODO_ |
| R3 | Sobreajuste | _TODO_ | _TODO_ |
| R4 | Sesgo por sexo (mayoría de hombres) | _TODO_ | _TODO_ |
| R5 | Limitaciones de cómputo del laboratorio de AWS (sesiones de 4 horas, presupuesto de $100, archivos fuera de `~/SageMaker` se pierden) | _TODO_ | _ej. push a GitHub en cada sesión, instancia pequeña y detenida al terminar, todo dentro de `~/SageMaker`_ |

---

## 6. Glosario

| Término | Definición |
|---|---|
| Accuracy | _TODO_ |
| Precision | _TODO_ |
| Recall | _TODO_ |
| F1‑Score | _TODO_ |
| Matriz de confusión | _TODO_ |
| Sobreajuste | _TODO_ |
| Fuga de información | _TODO_ |
| Modelo de referencia (baseline) | _TODO_ |
| Amazon SageMaker | _TODO_ |
| Instancia de notebook | _TODO_ |
| ... | ... |

## 7. Referencias

- Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). *Heart Disease* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X
- Pedregosa, F. et al. (2011). Scikit‑learn: Machine Learning in Python. *JMLR*, 12, 2825‑2830.
- Amazon Web Services. *Create an Amazon SageMaker notebook instance*. https://docs.aws.amazon.com/sagemaker/latest/dg/howitworks-create-ws.html
- _TODO: agregar las demás fuentes usadas (documentación, artículos sobre ética, C4, ADR)._
