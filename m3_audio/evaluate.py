#!/usr/bin/env python3
"""M3 — оценка распознавания речи.

Главная метрика модуля не WER сам по себе. Модуль существует ради того, чтобы
текст из голосового сообщения попал в M1, поэтому решающий вопрос такой:
меняют ли ошибки распознавания вердикт модерации?

Поэтому меряются обе величины:
  * WER / CER — насколько точна расшифровка;
  * согласованность вердиктов — доля записей, где решение M1 по расшифровке
    совпало с решением M1 по эталонному тексту.

Высокий WER при стопроцентной согласованности приемлем: модель может путать
окончания, но оскорбление остаётся оскорблением.

Запуск:
    python m3_audio/evaluate.py --sizes tiny base small
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "m1_text"))

from transcribe import SpeechRecognizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data"


def normalize(text: str) -> str:
    """Регистр и пунктуация на вердикт не влияют — убираем их перед сравнением."""
    import re

    text = text.lower().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest.csv"))
    ap.add_argument("--sizes", nargs="+", default=["small"],
                    help="какие размеры Whisper сравнить: tiny base small medium")
    ap.add_argument("--compute-type", default="int8")
    args = ap.parse_args()

    import jiwer
    from predict import TextModerator

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    print(f"записей: {len(rows)} | условия: "
          f"{sorted(set(r['condition'] for r in rows))}")

    # вердикты M1 по ЭТАЛОННЫМ текстам — с ними сравниваем
    moderator = TextModerator()
    reference_verdicts = {
        r["audio"]: moderator.decide(r["text"])["decision"] for r in rows
    }

    all_results = []
    for size in args.sizes:
        print(f"\n=== whisper-{size} / {args.compute_type} ===")
        rec = SpeechRecognizer(size, args.compute_type)
        print(f"загрузка {rec.load_s} с, +{rec.load_rss_mb} МБ")

        by_condition = defaultdict(lambda: {"refs": [], "hyps": [], "rtf": [], "agree": []})
        errors = []

        for row in rows:
            res = rec.transcribe(DATA / "audio" / row["audio"])
            ref, hyp = normalize(row["text"]), normalize(res["text"])
            cond = by_condition[row["condition"]]
            cond["refs"].append(ref)
            cond["hyps"].append(hyp)
            cond["rtf"].append(res["rtf"])

            verdict = moderator.decide(res["text"])["decision"]
            agrees = verdict == reference_verdicts[row["audio"]]
            cond["agree"].append(agrees)
            if not agrees:
                errors.append({
                    "audio": row["audio"],
                    "expected": reference_verdicts[row["audio"]],
                    "got": verdict,
                    "reference": row["text"],
                    "transcription": res["text"],
                })

        result = {"model": f"whisper-{size}", "compute_type": args.compute_type,
                  "load_s": rec.load_s, "load_rss_mb": rec.load_rss_mb,
                  "conditions": {}, "verdict_mismatches": errors}

        print(f"{'условие':>10} {'WER':>7} {'CER':>7} {'RTF':>7} {'вердикты':>10}")
        for cond_name, d in sorted(by_condition.items()):
            wer = jiwer.wer(d["refs"], d["hyps"])
            cer = jiwer.cer(d["refs"], d["hyps"])
            rtf = statistics.fmean(d["rtf"])
            agree = sum(d["agree"]) / len(d["agree"])
            print(f"{cond_name:>10} {wer:>7.3f} {cer:>7.3f} {rtf:>7.3f} {agree:>10.1%}")
            result["conditions"][cond_name] = {
                "n": len(d["refs"]),
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "rtf_mean": round(rtf, 3),
                "verdict_agreement": round(agree, 4),
            }

        if errors:
            print(f"\nрасхождения вердиктов ({len(errors)}):")
            for e in errors[:3]:
                print(f"  ожидалось {e['expected']}, получено {e['got']}")
                print(f"    эталон:  {e['reference']}")
                print(f"    распознано: {e['transcription']}")
        all_results.append(result)

    out = ROOT / "report" / "metrics" / "m3_audio.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nметрики →", out)


if __name__ == "__main__":
    main()
