from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.classification import split_dataset
from src.data import build_dataset, dataset_to_dataframe
from src.regression import (
    AgeRegressionGuide,
    evaluate_age_regressor,
    save_age_regressor,
    train_age_regressor,
)
from src.visualization import save_dataset_distribution_figure, save_regression_scatter


def _save_json(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_age_training_workflow(config: Any) -> dict[str, Any]:
    """Ejecuta el flujo completo de regresion de edad y devuelve un resumen."""

    config.ensure_output_dirs()

    print("[1/6] Cargando y preprocesando el dataset...")
    dataset = build_dataset(config)

    if len(dataset) == 0:
        raise RuntimeError(
            "No fue posible construir el dataset. Revise la ruta y el formato de nombres."
        )

    print(
        f"    Muestras validas: {len(dataset)} | Imagenes omitidas: {len(dataset.skipped_files)}"
    )

    dataset_df = dataset_to_dataframe(dataset)
    dataset_df.to_csv(config.reports_dir / "resumen_dataset.csv", index=False)
    save_dataset_distribution_figure(
        y_gender=dataset.y_gender,
        y_age=dataset.y_age,
        gender_map=config.gender_map,
        output_path=config.figures_dir / "distribucion_dataset.png",
    )

    print("[2/6] Separando entrenamiento y prueba...")
    split = split_dataset(
        dataset=dataset,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    print(f"    Train: {split.X_train.shape} | Test: {split.X_test.shape}")

    print("[3/6] Entrenando regresor de edad con PCA...")
    age_grid = train_age_regressor(
        X_train=split.X_train,
        y_age_train=split.y_age_train,
        pca_components=config.pca_components,
        random_state=config.random_state,
    )
    print(f"    Componentes PCA probados: {age_grid.param_grid['pca__n_components']}")
    print(f"    Mejor configuracion: {age_grid.best_params_}")

    print("[4/6] Evaluando regresor de edad...")
    age_eval = evaluate_age_regressor(
        model=age_grid.best_estimator_,
        X_test=split.X_test,
        y_age_test=split.y_age_test,
    )
    print(
        "    MAE={MAE:.3f} | RMSE={RMSE:.3f} | R2={R2:.3f}".format(**age_eval)
    )

    print("[5/6] Guardando artefactos...")
    save_age_regressor(
        model=age_grid.best_estimator_,
        output_path=config.models_dir / "pipeline_edad.pkl",
    )
    _save_json(age_eval, config.reports_dir / "metricas_edad.json")
    save_regression_scatter(
        y_true=split.y_age_test,
        y_pred=age_grid.best_estimator_.predict(split.X_test),
        output_path=config.figures_dir / "predicho_vs_real_edad.png",
    )

    print("[6/6] Generando guia breve...")
    guide = AgeRegressionGuide()
    (config.reports_dir / "guia_regresion.txt").write_text(
        guide.to_text(), encoding="utf-8"
    )

    summary = {
        "model_path": str(config.models_dir / "pipeline_edad.pkl"),
        "metrics_path": str(config.reports_dir / "metricas_edad.json"),
        "figure_path": str(config.figures_dir / "predicho_vs_real_edad.png"),
        "best_params": age_grid.best_params_,
        "metrics": age_eval,
        "samples": int(len(dataset)),
    }
    _save_json(summary, config.reports_dir / "resumen_entrenamiento_edad.json")
    return summary