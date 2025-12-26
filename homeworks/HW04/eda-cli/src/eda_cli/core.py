from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class ColumnSummary:
    name: str
    dtype: str
    non_null: int
    missing: int
    missing_share: float
    unique: int
    example_values: List[Any]
    is_numeric: bool
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    n_rows: int
    n_cols: int
    columns: List[ColumnSummary]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
        }


def _safe_float(value: Any) -> Optional[float]:
    """Аккуратно приводит значение к float (или возвращает None)."""
    if value is None:
        return None
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return None


def summarize_dataset(df: pd.DataFrame, example_values_per_column: int = 3) -> DatasetSummary:
    """
    Собирает сводку по датасету:
    - размеры таблицы;
    - по каждой колонке: тип, пропуски, уникальные, примеры;
    - для числовых — базовая статистика (min/max/mean/std).
    """
    n_rows, n_cols = df.shape
    summaries: List[ColumnSummary] = []

    for col_name in df.columns:
        series = df[col_name]
        dtype_str = str(series.dtype)

        not_null_count = int(series.notna().sum())
        missing_count = n_rows - not_null_count
        missing_share = float(missing_count / n_rows) if n_rows else 0.0

        uniq_count = int(series.nunique(dropna=True))

        # небольшая подборка примеров (как строки), чтобы удобнее печатать
        if not_null_count > 0:
            examples = (
                series.dropna()
                .astype(str)
                .unique()[:example_values_per_column]
                .tolist()
            )
        else:
            examples = []

        is_num = bool(ptypes.is_numeric_dtype(series))

        min_val = max_val = mean_val = std_val = None
        if is_num and not_null_count > 0:
            # pandas иногда возвращает numpy-скаляры — приводим к float
            min_val = _safe_float(series.min())
            max_val = _safe_float(series.max())
            mean_val = _safe_float(series.mean())
            std_val = _safe_float(series.std())

        summaries.append(
            ColumnSummary(
                name=col_name,
                dtype=dtype_str,
                non_null=not_null_count,
                missing=missing_count,
                missing_share=missing_share,
                unique=uniq_count,
                example_values=examples,
                is_numeric=is_num,
                min=min_val,
                max=max_val,
                mean=mean_val,
                std=std_val,
            )
        )

    return DatasetSummary(n_rows=n_rows, n_cols=n_cols, columns=summaries)


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает таблицу по пропускам:
    индекс = колонка, столбцы = missing_count / missing_share.
    """
    if df.empty:
        return pd.DataFrame(columns=["missing_count", "missing_share"])

    miss_cnt = df.isna().sum()
    miss_share = miss_cnt / len(df)

    out = pd.DataFrame(
        {"missing_count": miss_cnt, "missing_share": miss_share}
    ).sort_values("missing_share", ascending=False)

    return out


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Считает корреляцию Пирсона только по числовым колонкам.
    Если числовых колонок нет — возвращает пустой DataFrame.
    """
    num = df.select_dtypes(include="number")
    if num.empty:
        return pd.DataFrame()
    return num.corr(numeric_only=True)


def top_categories(
    df: pd.DataFrame,
    max_columns: int = 5,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для строковых/категориальных колонок формирует top-k значений.
    Результат: { column_name: DataFrame(value, count, share) }.
    """
    out: Dict[str, pd.DataFrame] = {}

    cat_cols: List[str] = []
    for col_name in df.columns:
        s = df[col_name]
        if ptypes.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
            cat_cols.append(col_name)

    for col_name in cat_cols[:max_columns]:
        s = df[col_name]
        vc = s.value_counts(dropna=True).head(top_k)
        if vc.empty:
            continue

        shares = (vc / vc.sum()).values
        table = pd.DataFrame(
            {
                "value": vc.index.astype(str),
                "count": vc.values,
                "share": shares,
            }
        )
        out[col_name] = table

    return out


def compute_quality_flags(
    df_or_summary,
    missing_df: Optional[pd.DataFrame] = None,
    df: Optional[pd.DataFrame] = None,
    *,
    zero_share_threshold: float = 0.9,
) -> Dict[str, Any]:
    """
    Простые эвристики качества данных.

    Поддерживает два режима вызова:
    1) compute_quality_flags(df, zero_share_threshold=0.9)
    2) compute_quality_flags(summary, missing_df, df, zero_share_threshold=0.9)

    Возвращает словарь флагов + агрегированную оценку quality_score.
    """
    # --- Нормализуем вход (чтобы прошли тесты и не сломался CLI) ---
    if isinstance(df_or_summary, pd.DataFrame):
        df_local = df_or_summary
        summary_local = summarize_dataset(df_local)
        missing_local = missing_table(df_local)
    else:
        # режим как в CLI: (summary, missing_df, df)
        summary_local = df_or_summary
        if df is None:
            raise TypeError("compute_quality_flags(): аргумент df обязателен в режиме (summary, missing_df, df)")
        df_local = df
        missing_local = missing_df if missing_df is not None else missing_table(df_local)

    flags: Dict[str, Any] = {}

    # --- базовые проверки размеров ---
    flags["too_few_rows"] = summary_local.n_rows < 100
    flags["too_many_columns"] = summary_local.n_cols > 100

    # --- пропуски ---
    max_missing_share = float(missing_local["missing_share"].max()) if not missing_local.empty else 0.0
    flags["max_missing_share"] = max_missing_share
    flags["too_many_missing"] = max_missing_share > 0.5

    # --- константные колонки ---
    constant_cols = [c for c in df_local.columns if df_local[c].nunique(dropna=True) <= 1]
    flags["has_constant_columns"] = len(constant_cols) > 0

    # --- много нулей в числовых колонках ---
    flags["has_many_zero_values"] = False
    for col in df_local.select_dtypes(include=["number"]).columns:
        s = df_local[col]
        total = len(s)
        if total == 0:
            continue
        zero_share = float((s == 0).sum() / total)
        if zero_share > zero_share_threshold:
            flags["has_many_zero_values"] = True
            break

    # --- quality_score (0..1) ---
    score = 1.0
    score -= max_missing_share
    if flags["too_few_rows"]:
        score -= 0.2
    if flags["too_many_columns"]:
        score -= 0.1
    if flags["has_constant_columns"]:
        score -= 0.15
    if flags["has_many_zero_values"]:
        score -= 0.1

    flags["quality_score"] = max(0.0, min(1.0, score))
    return flags


def flatten_summary_for_print(summary: DatasetSummary) -> pd.DataFrame:
    """
    Переводит DatasetSummary в плоскую таблицу (удобно печатать и сохранять в CSV).
    """
    rows: List[Dict[str, Any]] = []
    for col in summary.columns:
        rows.append(
            {
                "name": col.name,
                "dtype": col.dtype,
                "non_null": col.non_null,
                "missing": col.missing,
                "missing_share": col.missing_share,
                "unique": col.unique,
                "is_numeric": col.is_numeric,
                "min": col.min,
                "max": col.max,
                "mean": col.mean,
                "std": col.std,
            }
        )

    return pd.DataFrame(rows)
