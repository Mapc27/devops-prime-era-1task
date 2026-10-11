#!/usr/bin/env python3
"""M4 — оценка качества и обоснование шага сэмплирования.

Отвечает на три вопроса:
  1. Насколько верно модуль подтверждает заявленную категорию и ловит подмену.
  2. Какой шаг между анализируемыми кадрами выбрать: полный перебор точнее,
     но дороже. Свип по stride даёт кривую «качество ↔ время», по ней
     и выбирается рабочее значение.
  3. Что даёт MPS по сравнению с CPU на этой задаче.

Запуск:
    python m4_video/evaluate.py --models yolo11n.pt yolo11s.pt \
        --devices mps cpu --strides 1 5 10 15
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from detect_video import CONF, MIN_FRAMES, MODEL_ID, VideoInspector  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(__file__).parent / "data"
IMG_DIR = ROOT / "report" / "img"


def run_pass(insp: VideoInspector, rows: list[dict], stride: int) -> dict:
    """Один проход по всему манифесту с заданным шагом."""
    tp = fp = tn = fn = 0
    t0 = time.perf_counter()
    frames = 0
    # подтверждение верной категории в разрезе категорий: средняя accuracy
    # скрывает, что все ошибки могут приходиться на одну категорию
    per_category: dict[str, dict] = {}

    for row in rows:
        res = insp.inspect(
            DATA / "video" / row["video"],
            declared=row["declared"],
            stride=stride,
            conf=CONF,
            min_frames=MIN_FRAMES,
        )
        frames += res["frames_analyzed"]
        predicted = res["declared_present"]
        expected = row["should_pass"] == "1"

        if expected:
            cat = per_category.setdefault(row["true_category"], {"videos": 0, "confirmed": 0})
            cat["videos"] += 1
            cat["confirmed"] += int(predicted)

        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1  # честный товар не подтвердили
        elif not expected and predicted:
            fp += 1  # подмену пропустили
        else:
            tn += 1

    elapsed = time.perf_counter() - t0
    n = len(rows)
    return {
        "stride": stride,
        "accuracy": round((tp + tn) / n, 4),
        "confirm_recall": round(tp / (tp + fn), 4) if tp + fn else 0,
        "mismatch_recall": round(tn / (tn + fp), 4) if tn + fp else 0,
        "false_reject_rate": round(fn / (tp + fn), 4) if tp + fn else 0,
        "frames_analyzed": frames,
        "elapsed_s": round(elapsed, 2),
        "s_per_video": round(elapsed / n, 3),
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "per_category": per_category,
    }


def plot_sweep(sweep: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    strides = [s["stride"] for s in sweep]
    acc = [s["accuracy"] for s in sweep]
    sec = [s["elapsed_s"] for s in sweep]

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(strides, acc, "o-", color="tab:blue", label="accuracy")
    ax1.set_xlabel("шаг между анализируемыми кадрами (stride)")
    ax1.set_ylabel("accuracy", color="tab:blue")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(strides, sec, "s--", color="tab:red", label="время, с")
    ax2.set_ylabel("время полного прохода, с", color="tab:red")

    fig.suptitle("M4: качество против скорости при разном шаге сэмплирования")
    fig.tight_layout()
    out = IMG_DIR / "m4_stride_sweep.png"
    fig.savefig(out, dpi=130)
    print("график →", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(DATA / "manifest.csv"))
    ap.add_argument("--models", nargs="+", default=[MODEL_ID],
                    help="сравнить несколько размеров детектора, например: "
                         "yolo11n.pt yolo11s.pt")
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 5, 10, 15])
    ap.add_argument("--devices", nargs="+", default=["mps"], help="mps cpu")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
    print(f"проверок в манифесте: {len(rows)} | "
          f"роликов: {len(set(r['video'] for r in rows))}")

    metrics: dict = {"n_checks": len(rows), "models": {}}

    for model in args.models:
        metrics["models"][model] = {"devices": {}}
        for device in args.devices:
            insp = VideoInspector(model, device=device)
            sweep = []
            print(f"\n=== {model} на {device} ===")
            print(f"{'stride':>7} {'accuracy':>9} {'подмены':>9} {'ложн.откл':>10} "
                  f"{'кадров':>8} {'время,с':>9}")
            for stride in args.strides:
                r = run_pass(insp, rows, stride)
                sweep.append(r)
                print(f"{r['stride']:>7} {r['accuracy']:>9.3f} {r['mismatch_recall']:>9.3f} "
                      f"{r['false_reject_rate']:>10.3f} {r['frames_analyzed']:>8} "
                      f"{r['elapsed_s']:>9.2f}")
            metrics["models"][model]["devices"][device] = sweep
            # график строится по первой связке модель+устройство
            if model == args.models[0] and device == args.devices[0]:
                plot_sweep(sweep)

        if len(args.devices) > 1:
            base, alt = args.devices[0], args.devices[1]
            d = metrics["models"][model]["devices"]
            speedup = round(d[alt][0]["elapsed_s"] / d[base][0]["elapsed_s"], 2)
            metrics["models"][model]["speedup"] = {f"{base}_vs_{alt}": speedup}
            print(f"\n{model}: ускорение {base} против {alt} — {speedup}x")

    if len(args.models) > 1:
        print(f"\n{'модель':>14} {'accuracy':>9} {'время,с':>9}  (stride=5)")
        idx = args.strides.index(5) if 5 in args.strides else 0
        for model in args.models:
            r = metrics["models"][model]["devices"][args.devices[0]][idx]
            print(f"{model:>14} {r['accuracy']:>9.3f} {r['elapsed_s']:>9.2f}")

    out = ROOT / "report" / "metrics" / "m4_video.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("метрики →", out)


if __name__ == "__main__":
    main()
