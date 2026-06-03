from __future__ import annotations

import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from src.preprocessing import detect_faces, preprocess_face_array
except ImportError:
    from src.preprocessing import preprocess_face_array

    def detect_faces(
        image_bgr: np.ndarray,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (40, 40),
    ) -> np.ndarray:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(str(cascade_path))
        if detector.empty():
            try:
                detector = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_default.xml"))
            except Exception:
                detector = cv2.CascadeClassifier()

        if detector.empty():
            return np.empty((0, 4), dtype=int)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(faces) == 0:
            return np.empty((0, 4), dtype=int)
        return faces

GENDER_MAP = {0: "Mujer", 1: "Hombre"}
MODEL_DIRS = [
    REPO_ROOT / "artifacts" / "models",
    REPO_ROOT / "artifacts_clean_test" / "models",
]
GENDER_MODEL_NAME = "pipeline_genero.pkl"
AGE_MODEL_NAME = "pipeline_edad.pkl"


def _candidate_model_paths(model_name: str) -> list[Path]:
    return [directory / model_name for directory in MODEL_DIRS]


def _first_existing_path(model_name: str) -> Path | None:
    for candidate in _candidate_model_paths(model_name):
        if candidate.exists():
            return candidate
    return None


@st.cache_resource(show_spinner=False)
def load_models() -> tuple[object | None, Path | None, object | None, Path | None]:
    gender_path = _first_existing_path(GENDER_MODEL_NAME)
    age_path = _first_existing_path(AGE_MODEL_NAME)
    gender_model = joblib.load(gender_path) if gender_path is not None else None
    age_model = joblib.load(age_path) if age_path is not None else None
    return gender_model, gender_path, age_model, age_path


def _decode_uploaded_image(uploaded_file: object) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen subida.")
    return image


def _resize_for_display(image_bgr: np.ndarray, max_width: int = 520) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    if width <= max_width:
        return image_bgr
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_AREA)


def _predict_on_face(face_bgr: np.ndarray, gender_model: object, age_model: object) -> dict[str, object]:
    features, _ = preprocess_face_array(face_bgr, size=(25, 25))
    gender_raw = int(gender_model.predict([features])[0])
    age_raw = float(age_model.predict([features])[0])
    return {
        "gender_raw": gender_raw,
        "gender": GENDER_MAP.get(gender_raw, str(gender_raw)),
        "age": round(float(np.clip(age_raw, 0, 116)), 1),
        "age_raw": round(age_raw, 2),
    }


def _annotate_faces(image_bgr: np.ndarray, faces: np.ndarray, gender_model: object, age_model: object) -> tuple[np.ndarray, pd.DataFrame]:
    annotated = image_bgr.copy()
    rows: list[dict[str, object]] = []

    for idx, (x, y, w, h) in enumerate(faces, start=1):
        face = image_bgr[y : y + h, x : x + w]
        prediction = _predict_on_face(face, gender_model, age_model)
        label = f'{prediction["gender"]} | {prediction["age"]} años'

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.rectangle(annotated, (x, max(0, y - 24)), (x + max(140, w), y), (34, 197, 94), -1)
        cv2.putText(
            annotated,
            label,
            (x + 4, max(17, y - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        rows.append(
            {
                "rostro": idx,
                "género": prediction["gender"],
                "edad estimada": prediction["age"],
                "edad cruda": prediction["age_raw"],
                "bbox": f"{x},{y},{w},{h}",
            }
        )

    return annotated, pd.DataFrame(rows)


def run_app() -> None:
    st.set_page_config(page_title="UTKFace - Género y Edad", page_icon="🙂", layout="wide")

    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(180deg, #f7f4ef 0%, #ffffff 35%, #f3f0ea 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        .hero {
            padding: 1.4rem 1.5rem;
            border-radius: 1.2rem;
            background: linear-gradient(135deg, #1f2937 0%, #374151 55%, #4b5563 100%);
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.12);
        }
        .card {
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 1rem;
            padding: 1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.05);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin: 0;">UTKFace: género y edad</h1>
            <p style="margin: 0.4rem 0 0 0; font-size: 1.02rem; opacity: 0.92;">
                Sube una imagen con rostro y la app devolverá género y una edad estimada usando solo la cara detectada.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gender_model, gender_path, age_model, age_path = load_models()
    missing_models: list[str] = []
    if gender_model is None:
        missing_models.append("género")
    if age_model is None:
        missing_models.append("edad")

    if missing_models:
        st.error(
            "No se encontraron los modelos de "
            + ", ".join(missing_models)
            + ". Verifica que existan en artifacts/models."
        )
        st.stop()

    st.caption(
        f"Modelos cargados desde: {gender_path} y {age_path}"
    )

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is None:
        st.info("Sube una imagen para obtener la predicción de género y edad.")
        st.stop()

    image_bgr = _decode_uploaded_image(uploaded_file)
    faces = detect_faces(image_bgr)

    if len(faces) == 0:
        st.warning("No se detectó un rostro. Se usará la imagen completa como respaldo.")
        h, w = image_bgr.shape[:2]
        faces = np.array([[0, 0, w, h]])

    annotated, results_df = _annotate_faces(image_bgr, faces, gender_model, age_model)
    preview_original = _resize_for_display(image_bgr, max_width=520)
    preview_annotated = _resize_for_display(annotated, max_width=520)

    left, right = st.columns(2)

    with left:
        st.subheader("Imagen original")
        st.image(cv2.cvtColor(preview_original, cv2.COLOR_BGR2RGB), use_container_width=True)

    with right:
        st.subheader("Resultado")
        st.image(cv2.cvtColor(preview_annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Predicciones")
    st.dataframe(results_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    run_app()
