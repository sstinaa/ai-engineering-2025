from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .core import (
    DatasetSummary,
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)
from .viz import (
    plot_correlation_heatmap,
    plot_missing_matrix,
    plot_histograms_per_column,
    save_top_categories_tables,
)

app = typer.Typer(help="Небольшой CLI для EDA CSV-таблиц")


def _load_csv(path: Path, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Читает CSV и превращает ошибки чтения в понятные сообщения CLI."""
    if not path.exists():
        raise typer.BadParameter(f"Файл '{path}' не найден")

    try:
        return pd.read_csv(path, sep=sep, encoding=encoding)
    except Exception as exc:  # noqa: BLE001
        raise typer.BadParameter(f"Не получилось прочитать CSV: {exc}") from exc


@app.command()
def overview(
    path: str = typer.Argument(..., help="Путь к CSV."),
    sep: str = typer.Option(",", help="Разделитель."),
    encoding: str = typer.Option("utf-8", help="Кодировка."),
) -> None:
    """
    Показать общую сводку по датасету:
    - размеры (строки/столбцы);
    - типы данных;
    - компактный summary по колонкам.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)
    summary: DatasetSummary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)

    typer.echo(f"Строк: {summary.n_rows}")
    typer.echo(f"Столбцов: {summary.n_cols}")
    typer.echo("\nКолонки:")
    typer.echo(summary_df.to_string(index=False))


@app.command()
def head(
    path: str = typer.Argument(..., help="Путь к CSV."),
    n: int = typer.Option(5, help="Сколько первых строк показать."),
    sep: str = typer.Option(",", help="Разделитель."),
    encoding: str = typer.Option("utf-8", help="Кодировка."),
) -> None:
    """
    Вывести первые N строк из CSV.

    По смыслу аналогично unix `head` и `df.head()` в pandas —
    удобно для быстрого взгляда на структуру таблицы.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    if n <= 0:
        typer.echo("Параметр --n должен быть больше 0", err=True)
        raise typer.Exit(1)

    out = df.head(n)
    shown = min(n, len(df))

    typer.echo(
        f"Первые {shown} строк из {len(df)} (столбцов: {len(df.columns)}):\n"
    )
    typer.echo(out.to_string(index=True))


@app.command()
def sample(
    path: str = typer.Argument(..., help="Путь к CSV."),
    n: int = typer.Option(10, help="Размер случайной выборки."),
    sep: str = typer.Option(",", help="Разделитель."),
    encoding: str = typer.Option("utf-8", help="Кодировка."),
    seed: int = typer.Option(42, help="Seed для воспроизводимости."),
) -> None:
    """
    Показать случайные N строк.

    Полезно, когда хочется посмотреть “живые” значения из разных частей таблицы.
    """
    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    if n <= 0:
        typer.echo("Параметр --n должен быть больше 0", err=True)
        raise typer.Exit(1)

    if n > len(df):
        typer.echo(
            f"Запрошено {n} строк, но доступно только {len(df)} — выводим все.",
            err=True,
        )
        out = df
    else:
        out = df.sample(n=n, random_state=seed)

    shown = min(n, len(df))
    typer.echo(
        f"Случайная выборка: {shown} строк (всего: {len(df)}, столбцов: {len(df.columns)}):\n"
    )
    typer.echo(out.to_string(index=True))


@app.command()
def report(
    path: str = typer.Argument(..., help="Путь к CSV."),
    out_dir: str = typer.Option("reports", help="Папка, куда сохранять отчёт."),
    sep: str = typer.Option(",", help="Разделитель."),
    encoding: str = typer.Option("utf-8", help="Кодировка."),
    max_hist_columns: int = typer.Option(
        6, help="Максимальное число числовых колонок для гистограмм."
    ),
    top_k_categories: int = typer.Option(
        5, help="Сколько top-значений сохранять для категориальных колонок."
    ),
    title: str = typer.Option("EDA-отчёт", help="Заголовок Markdown-отчёта."),
) -> None:
    """
    Собрать EDA-отчёт (таблицы + изображения):
    - summary по колонкам;
    - пропуски;
    - корреляция;
    - top-k категорий;
    - гистограммы, матрица пропусков, тепловая карта корреляции.
    """
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    df = _load_csv(Path(path), sep=sep, encoding=encoding)

    # 1) Базовые таблицы
    summary = summarize_dataset(df)
    summary_df = flatten_summary_for_print(summary)
    missing_df = missing_table(df)
    corr_df = correlation_matrix(df)
    top_cats = top_categories(df, top_k=top_k_categories)

    # 2) Эвристики качества
    quality_flags = compute_quality_flags(summary, missing_df, df)

    # 3) Сохранение результатов
    summary_df.to_csv(out_root / "summary.csv", index=False)

    if not missing_df.empty:
        missing_df.to_csv(out_root / "missing.csv", index=True)

    if not corr_df.empty:
        corr_df.to_csv(out_root / "correlation.csv", index=True)

    save_top_categories_tables(top_cats, out_root / "top_categories")

    # 4) Markdown-отчёт
    md_path = out_root / "report.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"Источник: `{Path(path).name}`\n\n")
        f.write(f"Строк: **{summary.n_rows}**, столбцов: **{summary.n_cols}**\n\n")

        f.write("## Параметры генерации\n\n")
        f.write(f"- Гистограммы (макс.): **{max_hist_columns}**\n")
        f.write(f"- Top-K категорий: **{top_k_categories}**\n\n")

        f.write("## Эвристики качества\n\n")
        f.write(f"- Итоговая оценка: **{quality_flags['quality_score']:.2f}**\n")
        f.write(
            f"- Максимальная доля пропусков: **{quality_flags['max_missing_share']:.2%}**\n"
        )
        f.write(f"- Слишком мало строк: **{quality_flags['too_few_rows']}**\n")
        f.write(f"- Слишком много колонок: **{quality_flags['too_many_columns']}**\n")
        f.write(f"- Слишком много пропусков: **{quality_flags['too_many_missing']}**\n")
        f.write(f"- Константные колонки: **{quality_flags['has_constant_columns']}**\n")
        f.write(f"- Много нулей: **{quality_flags['has_many_zero_values']}**\n\n")

        f.write("## Колонки\n\n")
        f.write("Подробности — в `summary.csv`.\n\n")

        f.write("## Пропуски\n\n")
        if missing_df.empty:
            f.write("Пропуски не обнаружены (или данных нет).\n\n")
        else:
            f.write("См. `missing.csv` и `missing_matrix.png`.\n\n")

        f.write("## Корреляция числовых признаков\n\n")
        if corr_df.empty:
            f.write("Недостаточно числовых колонок для расчёта корреляции.\n\n")
        else:
            f.write("См. `correlation.csv` и `correlation_heatmap.png`.\n\n")

        f.write("## Категориальные признаки\n\n")
        if not top_cats:
            f.write("Категориальные (строковые) признаки не найдены.\n\n")
        else:
            f.write(f"Сохранены top-{top_k_categories} значений по каждой колонке.\n")
            f.write("Файлы лежат в `top_categories/`.\n\n")

        f.write("## Гистограммы\n\n")
        f.write(f"Построено до {max_hist_columns} гистограмм. См. `hist_*.png`.\n")

    # 5) Картинки
    plot_histograms_per_column(df, out_root, max_columns=max_hist_columns)
    plot_missing_matrix(df, out_root / "missing_matrix.png")
    plot_correlation_heatmap(df, out_root / "correlation_heatmap.png")

    typer.echo(f"Отчёт сохранён в: {out_root}")
    typer.echo(f"- Markdown: {md_path}")
    typer.echo("- Таблицы: summary.csv, missing.csv, correlation.csv, top_categories/*.csv")
    typer.echo("- Графики: hist_*.png, missing_matrix.png, correlation_heatmap.png")


if __name__ == "__main__":
    app()
