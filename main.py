"""Console menu for the Lightweight Kazakh Near-Duplicate Detector.

Run from the project root:  python main.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data import convert_csv_to_parquet, run_eda
from src.leakage import run_leakage_check

if hasattr(sys.stdout, "reconfigure"):  # Russian output on Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")

MENU = """
==============================================================
 Lightweight Kazakh Near-Duplicate Detector
==============================================================
 1 - Конвертировать CSV в Parquet (по частям, chunksize)
 2 - EDA: типы дубликатов, длины, пропуски, повторы
 3 - Групповое разбиение по исходному документу (без утечки)
 4 - Проверка утечки train/test (точные и почти-дубликаты)
 5 - Дообучить compact-трансформер (e5-small, 1 эпоха)
 6 - Сравнить методы на test и сохранить в SQLite
 7 - SQL-отчёт (10 запросов) -> pandas, сохранить CSV
 8 - Ввести два текста -> схожесть и тип дубликата
 0 - Выход
--------------------------------------------------------------"""


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "0"


def _print_json_like(title: str, data: dict) -> None:
    print(f"\n{title}")
    for key, value in data.items():
        print(f"  {key}: {value}")


def action_convert() -> None:
    summary = convert_csv_to_parquet()
    print(f"Записано строк по сплитам: {summary}")
    print(f"Parquet: {config.PROCESSED_DIR}")


def action_eda() -> None:
    summary = run_eda()
    _print_json_like("Общий размер и сплиты", {"total_rows": summary["total_rows"], "splits": summary["split_counts"]})
    _print_json_like("Классы дубликатов", summary["type_counts"])
    _print_json_like("Длина content (токены)", summary["content_len_tokens"])
    _print_json_like("Длина modified_content (токены)", summary["modified_len_tokens"])
    _print_json_like(
        "Качество данных",
        {
            "missing_values": summary["missing_values"] or "нет",
            "rows_with_repeated_content": summary["rows_with_repeated_content"],
            "identical_content_pairs": summary["identical_content_pairs"],
            "unique_original_documents": summary["unique_original_documents"],
        },
    )
    print(f"\nГрафик: {config.RESULTS_DIR / 'duplicate_types.png'}")


def action_grouped_split() -> None:
    from src.splits import build_grouped_split

    summary = build_grouped_split()
    _print_json_like("Групповое разбиение (по исходному документу)", summary["splits"])
    _print_json_like("Пересечение исходных документов между сплитами", summary["shared_original_documents"])


def action_leakage() -> None:
    report = run_leakage_check()
    _print_json_like("Проверка утечки train -> test", report)


def action_finetune() -> None:
    from src.model import finetune_encoder
    from src.data import load_split

    print("Загружаю train (это займёт время)...")
    train = load_split(config.TRAIN_SPLIT, ["id", "content", "modified_content", "similarity_score"])
    out = finetune_encoder(train)
    print(f"Модель сохранена: {out}")


def action_benchmark() -> None:
    from src.evaluate import run_benchmark
    from src.storage import connect, save_experiment, save_pairs

    bench = run_benchmark(log=print)
    conn = connect()
    try:
        save_pairs(conn, bench["test"])
        for metrics in bench["results"]:
            save_experiment(conn, metrics, bench["predictions"][metrics["method"]])
    finally:
        conn.close()
    print(f"\nСохранено в SQLite: {config.DB_PATH}")
    print(pd.DataFrame(bench["results"]).round(4).to_string(index=False))


def action_sql_report() -> None:
    from src.sql_queries import QUERIES
    from src.storage import connect, run_query

    method = _ask(f"Метод для запросов (Enter = {config.FINETUNED_METHOD}): ") or config.FINETUNED_METHOD
    out_dir = config.RESULTS_DIR / "sql"
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = connect()
    try:
        for name, spec in QUERIES.items():
            df = run_query(conn, name, method=method)
            df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")
            print(f"\n{spec['title']}")
            print(df.head(12).to_string(index=False) if not df.empty else "  (нет данных)")
    finally:
        conn.close()
    print(f"\nCSV-файлы: {out_dir}")


def _classify_pair(text_a: str, text_b: str) -> None:
    import joblib
    from src.evaluate import FINETUNED
    from src.model import embed, load_encoder, pair_features

    clf_path = config.MODELS_DIR / "type_clf_e5_finetuned.joblib"
    if not config.FINETUNED_MODEL_DIR.exists() or not clf_path.exists():
        print("Сначала выполните пункт 4 и 5: нужна дообученная модель и классификатор типов.")
        return
    model = load_encoder(config.FINETUNED_MODEL_DIR)
    clf = joblib.load(clf_path)
    ea = embed(model, [text_a])
    eb = embed(model, [text_b])
    score = float((ea * eb).sum())
    pred_type = clf.predict(pair_features(ea, eb))[0]
    print(f"\nМодель: {FINETUNED}")
    print(f"Оценка схожести (cosine): {score:.4f}")
    print(f"Предсказанный тип дубликата: {pred_type}")


def action_predict() -> None:
    print("Введите два текста (одна строка каждый). Пустая строка — назад.")
    text_a = _ask("Текст 1: ")
    if not text_a:
        return
    text_b = _ask("Текст 2: ")
    if not text_b:
        return
    _classify_pair(text_a, text_b)


ACTIONS = {
    "1": action_convert,
    "2": action_eda,
    "3": action_grouped_split,
    "4": action_leakage,
    "5": action_finetune,
    "6": action_benchmark,
    "7": action_sql_report,
    "8": action_predict,
}


def main() -> int:
    while True:
        print(MENU)
        choice = _ask("Выберите пункт: ")
        if choice in ("0", "q", "exit"):
            print("Выход.")
            return 0
        action = ACTIONS.get(choice)
        if action is None:
            print("Неизвестный пункт меню, попробуйте снова.")
            continue
        try:
            action()
        except FileNotFoundError as err:
            print(f"Файл не найден: {err}")
        except (ValueError, KeyError) as err:
            print(f"Ошибка данных: {err}")
        except KeyboardInterrupt:
            print("\nОперация прервана пользователем.")
        except Exception:  # keep the menu alive on unexpected errors
            print("Непредвиденная ошибка:")
            traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
