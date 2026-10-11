#!/usr/bin/env python3
"""M3 — фактическое отношение сигнал/шум в зашумлённых записях.

make_testset.py задаёт шум амплитудой генератора, а не в децибелах, поэтому
названия условий noisy15/noisy10 — условные. Здесь SNR измеряется по факту:
зашумлённая запись раскладывается на масштабированную чистую речь и остаток
(метод наименьших квадратов), SNR = 10·log10(мощность речи / мощность остатка).

Запуск:
    python m3_audio/measure_snr.py
"""

from __future__ import annotations

import csv
import json
import statistics
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
        assert w.getsampwidth() == 2, "ожидается 16-битный PCM"
    return np.frombuffer(raw, dtype=np.int16).astype(np.float64)


def snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    n = min(len(clean), len(noisy))
    c, x = clean[:n], noisy[:n]
    k = float(x @ c / (c @ c))  # amix при смешивании меняет громкость речи
    residual = x - k * c
    return 10 * np.log10(((k * c) ** 2).sum() / (residual ** 2).sum())


def main() -> None:
    rows = list(csv.DictReader((DATA / "manifest.csv").open(encoding="utf-8")))
    by_cond: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["condition"] == "clean":
            continue
        clean = read_wav(DATA / "audio" / f"{row['id']}_clean.wav")
        noisy = read_wav(DATA / "audio" / row["audio"])
        by_cond[row["condition"]].append(snr_db(clean, noisy))

    out = {}
    for cond, values in sorted(by_cond.items()):
        out[cond] = {
            "n": len(values),
            "snr_db_mean": round(statistics.fmean(values), 1),
            "snr_db_min": round(min(values), 1),
            "snr_db_max": round(max(values), 1),
        }
        print(f"{cond:>8}: SNR {out[cond]['snr_db_mean']} дБ "
              f"(от {out[cond]['snr_db_min']} до {out[cond]['snr_db_max']})")

    path = ROOT / "report" / "metrics" / "m3_snr.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", path)


if __name__ == "__main__":
    main()
