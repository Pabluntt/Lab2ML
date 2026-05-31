from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
GENDER_MODEL_PATH = BASE_DIR / "pipeline_genero.pkl"
AGE_MODEL_PATH = BASE_DIR / "pipeline_edad.pkl"

GENDER_MAP = {0: "Mujer", 1: "Hombre"}


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


@st.cache_resource
def load_models():
    gender_model = joblib.load(GENDER_MODEL_PATH)
    age_model = joblib.load(AGE_MODEL_PATH)
    return gender_model, age_model


def preprocess_face_array(face_bgr_or_gray, size=(25, 25)):
    if face_bgr_or_gray is None:
        raise ValueError("La imagen recibida es None.")

    if len(face_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_bgr_or_gray.copy()

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)

    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.38), int(h * 0.48))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)

    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_norm = gray_masked.astype(np.float32) / 255.0
    return gray_norm.flatten()


def detect_faces(image_bgr):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        try:
            detector = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_default.xml"))
        except Exception:
            detector = cv2.CascadeClassifier()

    if detector.empty():
        return np.array([])

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    return faces


def annotate_image(image_bgr, detections, gender_model, age_model):
    annotated = image_bgr.copy()
    results = []

    for idx, (x, y, w, h) in enumerate(detections, start=1):
        face = image_bgr[y : y + h, x : x + w]
        x_vec = preprocess_face_array(face)

        pred_gender = int(gender_model.predict([x_vec])[0])
        pred_age_raw = float(age_model.predict([x_vec])[0])
        pred_age = int(np.clip(round(pred_age_raw), 0, 116))

        label_gender = GENDER_MAP.get(pred_gender, str(pred_gender))
        label = f"{label_gender} | {pred_age} años"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (34, 197, 94), 2)
        cv2.rectangle(annotated, (x, max(0, y - 22)), (x + max(120, w), y), (34, 197, 94), -1)
        cv2.putText(
            annotated,
            label,
            (x + 4, max(16, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (10, 10, 10),
            1,
            cv2.LINE_AA,
        )

        results.append(
            {
                "rostro": idx,
                "genero_predicho": label_gender,
                "edad_predicha": pred_age,
                "edad_cruda": round(pred_age_raw, 2),
            }
        )

    return annotated, results


st.markdown(
    """
    <div class="hero">
        <h1 style="margin: 0;">UTKFace: género y edad</h1>
        <p style="margin: 0.4rem 0 0 0; font-size: 1.02rem; opacity: 0.92;">
            Sube una imagen con un rostro y la app devolverá la clase de género y una edad estimada.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Esta app reutiliza los modelos guardados por el notebook: pipeline_genero.pkl y pipeline_edad.pkl.")

try:
    gender_model, age_model = load_models()
except Exception as exc:
    st.error(f"No se pudieron cargar los modelos. Verifica que existan los archivos .pkl en la raíz del proyecto. Detalle: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("No se pudo leer la imagen subida.")
        st.stop()

    faces = detect_faces(image_bgr)

    if len(faces) == 0:
        st.warning("No se pudo cargar o no se detectó un rostro. Se usará la imagen completa como entrada.")
        h, w = image_bgr.shape[:2]
        faces = np.array([[0, 0, w, h]])

    annotated, results = annotate_image(image_bgr, faces, gender_model, age_model)

    left, right = st.columns(2)

    with left:
        st.subheader("Imagen original")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

    with right:
        st.subheader("Resultado")
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Predicciones")
    st.dataframe(results, use_container_width=True, hide_index=True)
else:
    st.info("Sube una imagen para obtener la predicción de género y edad.")
