from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def build_oval_mask(shape: tuple[int, int]) -> np.ndarray:
    """Construye una mascara oval simple para atenuar el fondo del rostro."""

    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (max(1, int(width * 0.38)), max(1, int(height * 0.48)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def preprocess_face_array(
    face_bgr_or_gray: np.ndarray,
    size: tuple[int, int] = (25, 25),
) -> tuple[np.ndarray, np.ndarray]:
    """Transforma un rostro en un vector numerico listo para el modelo.

    Pasos:
    1. convertir a grises si hace falta,
    2. redimensionar,
    3. ecualizar histograma,
    4. aplicar mascara oval,
    5. normalizar en [0, 1],
    6. aplanar.
    """

    if face_bgr_or_gray is None:
        raise ValueError("La imagen recibida es None.")

    if face_bgr_or_gray.ndim == 3:
        gray = cv2.cvtColor(face_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    elif face_bgr_or_gray.ndim == 2:
        gray = face_bgr_or_gray.copy()
    else:
        raise ValueError("La imagen debe ser 2D o 3D.")

    gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)

    mask = build_oval_mask(gray.shape)
    gray_masked = cv2.bitwise_and(gray, gray, mask=mask)
    gray_norm = gray_masked.astype(np.float32) / 255.0

    return gray_norm.flatten(), gray_norm


def preprocess_image_path(
    image_path: str | Path,
    size: tuple[int, int] = (25, 25),
) -> tuple[np.ndarray, np.ndarray]:
    """Lee una imagen desde disco y aplica el mismo preprocesamiento facial."""

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")
    return preprocess_face_array(image, size=size)


def detect_faces(
    image_bgr: np.ndarray,
    scale_factor: float = 1.05,
    min_neighbors: int = 3,
    min_size: tuple[int, int] = (30, 30),
) -> np.ndarray:
    """Detecta rostros con el clasificador Haar frontal de OpenCV."""

    detector = cv2.CascadeClassifier()

    local_cascade = str(Path(__file__).resolve().parents[1] / "haarcascade_frontalface_default.xml")
    if detector.load(local_cascade):
        pass
    else:
        system_cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if not detector.load(str(system_cascade)):
            try:
                detector.load(cv2.samples.findFile("haarcascade_frontalface_default.xml"))
            except Exception:
                pass

    if detector.empty():
        return np.empty((0, 4), dtype=int)

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)

    attempts: list[tuple[np.ndarray, float]] = [
        (gray_equalized, 1.0),
        (cv2.resize(gray_equalized, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC), 1.25),
        (cv2.resize(gray_equalized, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC), 1.5),
        (gray, 1.0),
    ]

    for attempt_gray, resize_factor in attempts:
        adjusted_min_size = (
            max(16, int(min_size[0] * resize_factor)),
            max(16, int(min_size[1] * resize_factor)),
        )
        faces = detector.detectMultiScale(
            attempt_gray,
            scaleFactor=max(1.01, scale_factor),
            minNeighbors=min_neighbors,
            minSize=adjusted_min_size,
        )
        if len(faces) != 0:
            if resize_factor == 1.0:
                return faces

            scaled_faces = []
            for x, y, w, h in faces:
                scaled_faces.append(
                    [
                        int(round(x / resize_factor)),
                        int(round(y / resize_factor)),
                        int(round(w / resize_factor)),
                        int(round(h / resize_factor)),
                    ]
                )
            return np.asarray(scaled_faces, dtype=int)

    return np.empty((0, 4), dtype=int)
