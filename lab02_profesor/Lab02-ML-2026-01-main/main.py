from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import LabConfig


def parse_args() -> argparse.Namespace:
    """Parsea los argumentos del orquestador principal."""

    parser = argparse.ArgumentParser(
        description=(
            "Laboratorio 02: clasificacion de genero con UTKFace. "
            "La regresion de edad queda como guia para estudiantes."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset"),
        help=(
            "Ruta a la carpeta que contiene las imagenes de UTKFace. "
            "Por defecto se usa ./dataset"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Carpeta donde se guardaran los modelos, reportes y figuras.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(25, 25),
        help="Tamano del rostro preprocesado. Ejemplo: --img-size 25 25",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Proporcion del conjunto de prueba.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para mantener resultados reproducibles.",
    )
    parser.add_argument(
        "--pca-components",
        type=int,
        nargs="+",
        default=[30, 50, 80, 100, 150, 200],
        help="Lista de componentes PCA a evaluar.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limite opcional para pruebas rapidas con menos imagenes.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> LabConfig:
    """Construye el objeto de configuracion del laboratorio."""

    return LabConfig(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        image_size=tuple(args.img_size),
        test_size=args.test_size,
        random_state=args.random_state,
        pca_components=tuple(args.pca_components),
        max_images=args.max_images,
    )


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    """Guarda un diccionario JSON de metricas en disco."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Ejecuta el flujo completo de regresion de edad."""

    args = parse_args()

    from src.training import run_age_training_workflow

    config = build_config(args)
    summary = run_age_training_workflow(config)

    print("Proceso completado.")
    print(f"Modelo guardado en: {summary['model_path']}")
    print(f"Reporte de metricas en: {summary['metrics_path']}")
    print("Revise tambien src/regression.py y src/streamlit_app.py.")


if __name__ == "__main__":
    main()
