#!/usr/bin/env python3
"""M3 — генерация эталонного тестового набора для измерения WER.

Проблема: чтобы посчитать WER, нужны записи с ТОЧНО известным текстом.
Размеченных публичных корпусов под домен «голосовые к объявлениям» нет.

Решение: синтезируем речь системным голосом macOS (say -v Milena) по заранее
заданным фразам. Эталонный текст известен по построению.

Ограничение, которое обязательно учитывать при чтении метрик: синтезированная
речь заметно чище живой (нет акцента, оговорок, дальнего микрофона), поэтому
WER на таком наборе — оптимистичная оценка, нижняя граница ошибки.
Чтобы приблизить условия к реальным, дополнительно генерируются версии
с наложенным белым шумом двух уровней. Уровень задаётся амплитудой генератора,
поэтому названия noisy15/noisy10 условные; фактический SNR (около 13 и 7 дБ)
измеряет measure_snr.py.

Использование:
    python m3_audio/make_testset.py              # чистые + шумные
    python m3_audio/make_testset.py --clean-only
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

DATA = Path(__file__).parent / "data"
AUDIO = DATA / "audio"
VOICE = "Milena"  # русский системный голос macOS
# амплитуда шума в миксе; имена условные, фактический SNR — measure_snr.py
NOISE_LEVELS = {"noisy15": 0.10, "noisy10": 0.20}


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def synth(text: str, out_wav: Path) -> None:
    """say выдаёт aiff, ffmpeg приводит к 16 кГц моно — формату входа Whisper."""
    aiff = out_wav.with_suffix(".aiff")
    run(["say", "-v", VOICE, "-o", str(aiff), text])
    run(["ffmpeg", "-y", "-i", str(aiff), "-ar", "16000", "-ac", "1", str(out_wav)])
    aiff.unlink()


def add_noise(src: Path, dst: Path, amplitude: float) -> None:
    run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-filter_complex",
            f"anoisesrc=a={amplitude}:c=white:d=60[n];"
            f"[0:a][n]amix=inputs=2:duration=first:weights=1 1[out]",
            "-map", "[out]", "-ar", "16000", "-ac", "1", str(dst),
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-only", action="store_true")
    args = ap.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader((DATA / "phrases.csv").open(encoding="utf-8")))
    manifest = []

    for row in rows:
        clean = AUDIO / f"{row['id']}_clean.wav"
        synth(row["text"], clean)
        manifest.append({**row, "audio": clean.name, "condition": "clean"})
        print(f"  {clean.name}")

        if args.clean_only:
            continue
        for cond, amp in NOISE_LEVELS.items():
            noisy = AUDIO / f"{row['id']}_{cond}.wav"
            add_noise(clean, noisy, amp)
            manifest.append({**row, "audio": noisy.name, "condition": cond})
            print(f"  {noisy.name}")

    out = DATA / "manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "label", "audio", "condition"])
        w.writeheader()
        w.writerows(manifest)

    print(f"\nсоздано записей: {len(manifest)} | манифест → {out}")


if __name__ == "__main__":
    main()
