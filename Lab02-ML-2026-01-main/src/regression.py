from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.classification import resolve_pca_components


@dataclass(slots=True)
class AgeRegressionGuide:
    """Resume los pasos que deben seguir los estudiantes para la regresion."""

    scoring: str = "neg_mean_absolute_error"
    suggested_metrics: tuple[str, ...] = ("MAE", "RMSE", "R2")

    def to_text(self) -> str:
        """Genera un recordatorio corto para acompanar el laboratorio."""

        return (
            "Guia de regresion de edad\n"
            "========================\n"
            "\n"
            "Se implemento un pipeline PCA + Ridge (regresion lineal regularizada).\n"
            "Ajuste: GridSearchCV con scoring=neg_mean_absolute_error.\n"
            "Metricas sugeridas: MAE, RMSE, R2.\n"
        )


def build_age_regression_pipeline(random_state: int) -> Any:
    """Construye el pipeline de regresion: PCA -> Ridge."""

    return Pipeline(
        [
            ("pca", PCA(whiten=True, random_state=random_state)),
            ("reg", Ridge(random_state=random_state)),
        ]
    )


def train_age_regressor(
    X_train: Any,
    y_age_train: Any,
    pca_components: tuple[int, ...],
    random_state: int,
    requested_cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 1,
) -> GridSearchCV:
    """Entrena un regresor de edad usando GridSearchCV sobre pca__n_components y reg__alpha.

    Retorna el objeto GridSearchCV ajustado.
    """

    n_samples = int(X_train.shape[0])
    cv_folds = max(2, min(requested_cv, n_samples))

    safe_components = resolve_pca_components(
        candidates=pca_components, X_train=X_train, cv_folds=cv_folds
    )

    pipeline = build_age_regression_pipeline(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid={
            "pca__n_components": safe_components,
            "reg__alpha": [0.1, 1, 10, 100],
        },
        scoring="neg_mean_absolute_error",
        cv=cv_folds,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    grid_search.fit(X_train, y_age_train)

    return grid_search


def evaluate_age_regressor(model: Any, X_test: Any, y_age_test: Any) -> dict[str, float]:
    """Calcula MAE, RMSE y R2 para el regresor dado."""

    y_pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_age_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_age_test, y_pred)))
    r2 = float(r2_score(y_age_test, y_pred))

    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def save_age_regressor(model: Any, output_path: str) -> None:
    """Guarda el pipeline completo en disco como joblib."""

    output_path = Path(output_path) if not isinstance(output_path, Path) else output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
