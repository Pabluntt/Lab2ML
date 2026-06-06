
# UTKFace — Clasificación de Género y Regresión de Edad

Pipeline de clasificación de género (PCA + GaussianNB) y regresión de edad (PCA + LinearRegression) sobre el dataset UTKFace, con interfaz visual en Streamlit.

## Requisitos

- Python 3.10+
- Las dependencias se listan en `requirements.txt`

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

## Entrenar modelos

```bash
cd Lab02-ML-2026-01-main
python main.py
```

Esto entrena ambos modelos y guarda:

- `artifacts/models/pipeline_genero.pkl`
- `artifacts/models/pipeline_edad.pkl`
- Figuras en `artifacts/figures/`
- Métricas en `artifacts/reports/`

## Ejecutar app (Streamlit)

```bash
cd Lab02-ML-2026-01-main
streamlit run main_visual.py
```

## Estructura

```
Lab02-ML-2026-01-main/
├── main.py              # Orquestador de entrenamiento
├── main_visual.py       # App Streamlit
├── src/
│   ├── config.py        # Parámetros globales
│   ├── data.py          # Carga y preprocesamiento
│   ├── preprocessing.py # Detección facial
│   ├── classification.py# Clasificador de género
│   ├── regression.py    # Regresor de edad
│   ├── training.py      # Flujos de entrenamiento
│   └── visualization.py # Generación de figuras
├── dataset/             # Imágenes UTKFace
└── artifacts/           # Modelos, figuras, reportes
```

Ver todos: `python main.py --help`
