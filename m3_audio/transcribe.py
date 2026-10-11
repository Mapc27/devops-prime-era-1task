#!/usr/bin/env python3
"""M3 — распознавание речи в голосовых комментариях к объявлению.

Расшифровка не самоцель: полученный текст передаётся в M1, иначе нарушения
в голосовых сообщениях не отлавливаются вообще. Поэтому главная метрика модуля
не WER сам по себе, а согласованность вердиктов — ошибки распознавания
не должны менять решение модерации.

Про устройство: CTranslate2 (движок faster-whisper) не поддерживает Metal,
поэтому модель работает на CPU. В режиме int8 на M4 Pro это даёт RTF около
0.27 — в 3-4 раза быстрее реального времени, так что ограничение не мешает.

Примеры:
    python m3_audio/transcribe.py --audio m3_audio/data/audio/01_clean.wav
    python m3_audio/transcribe.py --audio file.mp3 --srt --moderate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import rss_mb  # noqa: E402

MODEL_SIZE = "small"
COMPUTE_TYPE = "int8"


def format_timestamp(seconds: float) -> str:
    ms = int(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class SpeechRecognizer:
    def __init__(self, size: str = MODEL_SIZE, compute_type: str = COMPUTE_TYPE):
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        rss0 = rss_mb()
        self.size = size
        self.compute_type = compute_type
        self.model = WhisperModel(size, device="cpu", compute_type=compute_type)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.load_rss_mb = round(rss_mb() - rss0, 1)

    def transcribe(self, path: str | Path, language: str | None = "ru") -> dict:
        t0 = time.perf_counter()
        segments, info = self.model.transcribe(
            str(path),
            language=language,
            vad_filter=True,  # отсекает тишину, ускоряет и снижает галлюцинации
            beam_size=5,
        )
        # генератор -> список, только здесь реально считается модель
        segments = list(segments)
        elapsed = time.perf_counter() - t0

        return {
            "audio": Path(path).name,
            "text": " ".join(s.text.strip() for s in segments).strip(),
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration_s": round(info.duration, 2),
            "elapsed_s": round(elapsed, 2),
            # RTF < 1 означает, что обработка быстрее реального времени
            "rtf": round(elapsed / info.duration, 3) if info.duration else None,
            "segments": [
                {
                    "start": round(s.start, 2),
                    "end": round(s.end, 2),
                    "text": s.text.strip(),
                    "avg_logprob": round(s.avg_logprob, 3),
                    "no_speech_prob": round(s.no_speech_prob, 3),
                }
                for s in segments
            ],
        }

    @staticmethod
    def to_srt(segments: list[dict]) -> str:
        blocks = []
        for i, s in enumerate(segments, 1):
            blocks.append(
                f"{i}\n{format_timestamp(s['start'])} --> "
                f"{format_timestamp(s['end'])}\n{s['text']}\n"
            )
        return "\n".join(blocks)


def moderate_text(text: str) -> dict:
    """Передача расшифровки в M1 — ради этого модуль и существует."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m1_text"))
    from predict import TextModerator

    return TextModerator().decide(text)


def main() -> None:
    ap = argparse.ArgumentParser(description="Распознавание голосового комментария")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--size", default=MODEL_SIZE, help="tiny / base / small / medium")
    ap.add_argument("--compute-type", default=COMPUTE_TYPE, help="int8 / float32")
    ap.add_argument("--language", default="ru", help="или auto для автоопределения")
    ap.add_argument("--srt", action="store_true", help="сохранить субтитры")
    ap.add_argument("--moderate", action="store_true", help="прогнать текст через M1")
    args = ap.parse_args()

    rec = SpeechRecognizer(args.size, args.compute_type)
    print(f"модель {args.size}/{args.compute_type} загружена за {rec.load_s} с "
          f"(+{rec.load_rss_mb} МБ)")

    result = rec.transcribe(args.audio, None if args.language == "auto" else args.language)

    print(f"\nязык: {result['language']} ({result['language_probability']})")
    print(f"длительность {result['duration_s']} с | обработка {result['elapsed_s']} с "
          f"| RTF {result['rtf']}")
    print(f"\nтекст: {result['text']}")

    if args.srt:
        out = Path("m3_audio/out") / (Path(args.audio).stem + ".srt")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rec.to_srt(result["segments"]), encoding="utf-8")
        print("субтитры →", out)

    if args.moderate:
        verdict = moderate_text(result["text"])
        print("\nвердикт M1 по расшифровке:")
        print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
