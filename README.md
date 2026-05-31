# UTKFace - Género y Edad

Aplicación simple de Streamlit para predecir género y edad a partir de una imagen facial.

## Archivos importantes

- `app.py`: app principal de Streamlit.
- `pipeline_genero.pkl`: modelo entrenado para clasificación de género.
- `pipeline_edad.pkl`: modelo entrenado para regresión de edad.

## Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Nota

Si quieres mejorar las predicciones, vuelve a entrenar los pipelines desde `laboratorio_02_utkface_genero_edad_colab.ipynb` y vuelve a generar los archivos `.pkl`.