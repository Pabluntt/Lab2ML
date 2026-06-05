# Lab02-ML-2026-01

Código modular para el Laboratorio 02 de Machine Learning:

- clasificación de género con `GaussianNB` (PCA + GridSearchCV)
- regresión de edad con `Ridge` (PCA + GridSearchCV con regularización)
- despliegue completo con `Streamlit` (detección facial + predicción de género y edad)

## Estructura del proyecto

```text
.
├── dataset/                   # Imágenes UTKFace
├── artifacts/                 # Modelos, reportes y figuras (generados al entrenar)
│   ├── models/
│   ├── reports/
│   └── figures/
├── src/
│   ├── __init__.py
│   ├── config.py              # Parámetros globales
│   ├── data.py                # Carga y parseo del dataset
│   ├── preprocessing.py       # Preprocesamiento facial (gray, resize, ecualización, máscara oval)
│   ├── classification.py      # PCA + GaussianNB para género
│   ├── regression.py          # PCA + Ridge para edad
│   ├── training.py            # Flujos de entrenamiento (género + edad)
│   ├── inference.py           # Inferencia con modelos guardados
│   ├── visualization.py       # Figuras (distribución, matriz de confusión, PCA)
│   └── streamlit_app.py       # App Streamlit con detección facial y predicciones
├── main.py                    # Orquestador: entrena género + edad
├── main_visual.py             # Punto de entrada para Streamlit
├── requirements.txt           # Dependencias
├── enviroment.yml             # Entorno Conda
└── README.md
```

## Requisitos

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
# Entrenar ambos modelos (usa dataset/ por defecto)
python main.py

# Prueba rápida con menos imágenes
python main.py --max-images 5000
```

Esto genera:
- `artifacts/models/pipeline_genero.pkl` — clasificador de género
- `artifacts/models/pipeline_edad.pkl` — regresor de edad
- `artifacts/reports/metricas_genero.json` — accuracy, precisión, recall, F1
- `artifacts/reports/metricas_edad.json` — MAE, RMSE, R2
- `artifacts/figures/` — distribuciones, matriz de confusión, proyección PCA

## App visual

```bash
streamlit run main_visual.py
```

Funcionalidades:
- Subir imagen (jpg, jpeg, png)
- Detección facial automática con Haar Cascade
- Predicción de género (Hombre / Mujer)
- Predicción de edad estimada
- Panel de diagnóstico con número de rostros detectados

## Opciones de main.py

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--dataset-dir` | `dataset/` | Ruta al dataset UTKFace |
| `--output-dir` | `artifacts/` | Carpeta de salida |
| `--img-size` | `25 25` | Tamaño del rostro preprocesado |
| `--test-size` | `0.20` | Proporción de prueba |
| `--max-images` | `None` | Límite opcional para pruebas |
| `--pca-components` | `30 50 80 100 150 200` | Componentes PCA a evaluar |
| `--random-state` | `42` | Semilla reproducible |
