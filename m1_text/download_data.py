#!/usr/bin/env python3
"""M1 — подготовка публичного корпуса для измерения качества.

Берётся тестовый сплит открытого корпуса русских токсичных комментариев.

ВАЖНАЯ ОГОВОРКА, без которой метрики можно неверно прочитать: выбранная модель
обучалась на корпусах этого же происхождения, поэтому результат на публичных
данных — оценка сверху, а не оценка обобщения. Честная проверка переноса на наш
домен — на собственном наборе объявлений (data/samples.csv), который модель
заведомо не видела. Поэтому evaluate.py прогоняется на обоих наборах.

Запуск:
    python m1_text/download_data.py
    python m1_text/download_data.py --dataset mteb/RuToxicOKMLCUPClassification
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path(__file__).parent / "data"
DEFAULT = "AlexSham/Toxic_Russian_Comments"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 — сохранить весь сплит")
    ap.add_argument("--out", default=str(DATA / "public_test.csv"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit and args.limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(args.limit))

    df = ds.to_pandas()
    # разные корпуса называют колонку метки по-разному
    label_col = next((c for c in ("label", "toxic", "is_toxic") if c in df.columns), None)
    if label_col is None:
        raise SystemExit(f"не найдена колонка метки среди {list(df.columns)}")

    df = df[["text", label_col]].rename(columns={label_col: "label"})
    df["label"] = df["label"].astype(int)
    df = df[df["text"].astype(str).str.strip().str.len() > 0]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"источник: {args.dataset} [{args.split}]")
    print(f"сохранено: {len(df)} строк | нарушений: {df['label'].sum()} "
          f"({df['label'].mean():.1%})")
    print("→", out)


if __name__ == "__main__":
    main()
