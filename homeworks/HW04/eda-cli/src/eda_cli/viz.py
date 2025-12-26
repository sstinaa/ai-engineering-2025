from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import pandas as pd

PathLike = Union[str, Path]


def _ensure_dir(path: PathLike) -> Path:
    """Создать директорию (если нужно) и вернуть Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_histograms_per_column(
    df: pd.DataFrame,
    out_dir: PathLike,
    max_columns: int = 6,
    bins: int = 20,
) -> List[Path]:
    """
    Строит отдельные гистограммы для числовых признаков.
    Сохраняет PNG в out_dir и возвращает список путей.
    """
    out_root = _ensure_dir(out_dir)
    num = df.select_dtypes(include="number")

    saved: List[Path] = []
    for idx, col in enumerate(num.columns[:max_columns], start=1):
        values = num[col].dropna()
        if values.empty:
            continue

        fig, ax = plt.subplots()
        ax.hist(values.to_numpy(), bins=bins)
        ax.set_title(f"Histogram: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        fig.tight_layout()

        out_path = out_root / f"hist_{idx}_{col}.png"
        fig.savefig(out_path)
        plt.close(fig)

        saved.append(out_path)

    return saved


def plot_missing_matrix(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Визуализация пропусков в виде матрицы:
    True = пропуск, False = значение.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Dataset is empty", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    mask = df.isna().to_numpy()

    # размер подбираем по числу колонок, чтобы подписи помещались
    width = min(12.0, max(6.0, df.shape[1] * 0.45))
    fig, ax = plt.subplots(figsize=(width, 4.0))
    ax.imshow(mask, aspect="auto", interpolation="none")

    ax.set_title("Missing values overview")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    ax.set_xticks(range(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=90, fontsize=8)
    ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_path: PathLike) -> Path:
    """
    Сохраняет heatmap корреляции (Pearson) для числовых колонок.
    Если числовых колонок меньше двух — рисует информационную заглушку.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        fig, ax = plt.subplots()
        ax.text(
            0.5,
            0.5,
            "Correlation is unavailable (need 2+ numeric columns)",
            ha="center",
            va="center",
        )
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        return path

    corr = num.corr(numeric_only=True)

    fig_w = min(10.0, max(6.0, float(corr.shape[1])))
    fig_h = min(8.0, max(5.0, float(corr.shape[0]) * 0.8))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")

    ax.set_title("Correlation heatmap")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(corr.shape[0]))
    ax.set_yticklabels(corr.index, fontsize=8)

    fig.colorbar(im, ax=ax, label="Pearson correlation (r)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_top_categories_tables(
    top_cats: Dict[str, pd.DataFrame],
    out_dir: PathLike,
) -> List[Path]:
    """
    Сохраняет таблицы top-k категорий по каждой колонке в отдельные CSV.
    Возвращает список созданных файлов.
    """
    out_root = _ensure_dir(out_dir)

    created: List[Path] = []
    for col_name, table in top_cats.items():
        out_path = out_root / f"top_values_{col_name}.csv"
        table.to_csv(out_path, index=False)
        created.append(out_path)

    return created
