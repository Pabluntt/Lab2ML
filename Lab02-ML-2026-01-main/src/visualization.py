from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


def save_dataset_distribution_figure(
    y_gender: np.ndarray,
    y_age: np.ndarray,
    gender_map: dict[int, str],
    output_path: str | Path,
) -> None:
    """Guarda una figura simple con distribucion de genero y edad."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(int(label) for label in np.unique(y_gender))
    counts = [int(np.sum(y_gender == label)) for label in unique_labels]
    labels = [gender_map.get(label, str(label)) for label in unique_labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(labels, counts, color=["#4C78A8", "#F58518"][: len(labels)])
    axes[0].set_title("Distribucion de genero")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(y_age, bins=20, color="#54A24B", edgecolor="black")
    axes[1].set_title("Distribucion de edad")
    axes[1].set_xlabel("Edad")
    axes[1].set_ylabel("Frecuencia")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix_figure(
    confusion_matrix: np.ndarray,
    labels: list[str],
    output_path: str | Path,
) -> None:
    """Guarda la matriz de confusion del clasificador."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusion - genero")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_pca_projection_figure(
    X: np.ndarray,
    y: np.ndarray,
    gender_map: dict[int, str],
    random_state: int,
    output_path: str | Path,
) -> None:
    """Proyecta el conjunto de entrenamiento a 2D para una visualizacion simple."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    projector = PCA(n_components=2, random_state=random_state)
    projected = projector.fit_transform(X)

    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(int(value) for value in np.unique(y)):
        mask = y == label
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            label=gender_map.get(label, str(label)),
            alpha=0.65,
            s=18,
        )

    ax.set_title("Proyeccion PCA del conjunto de entrenamiento")
    ax.set_xlabel("Componente principal 1")
    ax.set_ylabel("Componente principal 2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_regression_scatter(
    y_true: np.ndarray, y_pred: np.ndarray, output_path: str | Path
) -> None:
    """Guarda un scatter de predicho vs real para la regresion de edad."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, s=18, color="#2b8cbe")
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Edad real")
    ax.set_ylabel("Edad predicha")
    ax.set_title("Predicho vs Real - Regresion de edad")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_gender_validation_curve_figure(
    grid_search: Any,
    output_path: str | Path,
) -> None:
    """Curva de validacion: score CV medio vs componentes PCA para genero."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv_results = grid_search.cv_results_
    components = [int(p) for p in cv_results["param_pca__n_components"]]
    mean_scores = cv_results["mean_test_score"]
    std_scores = cv_results["std_test_score"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(components, mean_scores, yerr=std_scores, fmt="-o", capsize=5, color="#4C78A8")
    ax.set_xlabel("Componentes PCA")
    ax.set_ylabel("F1-score (CV medio)")
    ax.set_title("Curva de validacion - Clasificador de genero")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_scree_plot_figure(
    X_train: np.ndarray,
    output_path: str | Path,
    n_components: int = 50,
) -> None:
    """Grafico de sedimentacion (scree plot) de la varianza explicada por PCA."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.bar(range(1, len(explained) + 1), explained, alpha=0.7, color="#54A24B", label="Varianza explicada individual")
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (proporcion)")
    ax1.set_title("Scree plot - Varianza explicada por componente PCA")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative) + 1), cumulative, "-o", color="#F58518", markersize=4, label="Varianza acumulada")
    ax2.set_ylabel("Varianza explicada acumulada")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
