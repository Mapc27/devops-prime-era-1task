#!/usr/bin/env python3
"""M2 — оценка качества и цена универсальности zero-shot подхода.

Три измерения:
  1. Точность zero-shot классификации по каталогу категорий.
  2. Сравнение с ОБУЧЕННЫМ классификатором (ResNet50, ImageNet) на тех же классах.
     Это и есть «цена универсальности»: сколько точности мы платим за то, что
     категорию можно добавить строкой в JSON, а не переобучением.
  3. Детекция несоответствия — основной рабочий сценарий модуля: пользователь
     заявил одну категорию, а на фото другое.

Запуск:
    python m2_image/download_data.py
    python m2_image/evaluate.py --data m2_image/data/imagenette_val --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.bench import Bench, get_device  # noqa: E402
from predict import MATCH_THRESHOLD, MODEL_ID, CategoryMatcher, decide  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"


def build_class_mapping(label_names: list[str]) -> dict[int, str]:
    """Класс датасета -> категория площадки (через source_class в каталоге)."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["categories"]
    # в тестовом датасете представлена только часть категорий каталога
    by_source = {
        v["eval_source_class"].lower(): k
        for k, v in catalog.items()
        if v.get("eval_source_class")
    }

    mapping = {}
    for idx, name in enumerate(label_names):
        key = name.lower().replace("_", " ").strip()
        if key in by_source:
            mapping[idx] = by_source[key]
            continue
        # запасной вариант: совпадение по подстроке
        hit = next((s for s in by_source if s in key or key in s), None)
        if hit is None:
            raise SystemExit(
                f"класс датасета «{name}» не найден в catalog.json.\n"
                f"известные source_class: {sorted(by_source)}"
            )
        mapping[idx] = by_source[hit]
    return mapping


# --------------------------------------------------------------------------- #
# обученный классификатор для сравнения
# --------------------------------------------------------------------------- #
class SupervisedBaseline:
    """ResNet50, обученный на ImageNet-1k.

    Знает ровно эти классы и ничего кроме них. Выход ограничивается
    десятью интересующими индексами — это даёт классификатору фору
    (задача из 1000 классов сводится к 10) и делает сравнение честным
    в пользу обученной модели.
    """

    def __init__(self, source_classes: list[str], device: str):
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=self.weights).to(device).eval()
        self.device = device
        self.preprocess = self.weights.transforms()

        names = [c.lower() for c in self.weights.meta["categories"]]
        self.indices, self.classes = [], []
        for cls in source_classes:
            key = cls.lower()
            idx = names.index(key) if key in names else next(
                (i for i, n in enumerate(names) if key in n or n in key), None
            )
            if idx is None:
                raise SystemExit(f"класс «{cls}» не найден среди классов ImageNet")
            self.indices.append(idx)
            self.classes.append(cls)

    def predict(self, images: list) -> list[str]:
        batch = self.torch.stack([self.preprocess(im) for im in images]).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(batch)[:, self.indices]
        return [self.classes[i] for i in logits.argmax(dim=-1).cpu().tolist()]


# --------------------------------------------------------------------------- #
def perf_sweep(images: list, model_id: str, devices: list[str],
               batches: list[int], n_images: int) -> dict:
    """Замер скорости отдельно от качества.

    Точность от устройства и размера батча не зависит — считается один раз.
    Здесь меряется только стоимость инференса: во сколько раз MPS выигрывает
    у CPU и окупается ли батчинг.
    """
    subset = images[:n_images]
    out: dict = {}

    for device in devices:
        matcher = CategoryMatcher(model_id, device=device, with_distractors=True)
        out[device] = {}
        print(f"\n=== скорость на {device} ({len(subset)} изображений) ===")
        print(f"{'батч':>6} {'мс/изображение':>16} {'изображений/с':>15}")

        for batch in batches:
            # прогрев: первый вызов на MPS включает компиляцию ядер
            matcher.classify(subset[: min(batch, len(subset))])

            bench = Bench(f"m2_perf_{device}_b{batch}", device=device,
                          extra={"batch": batch})
            with bench:
                for i in range(0, len(subset), batch):
                    with bench.step():
                        matcher.classify(subset[i : i + batch])

            summary = bench.summary()
            per_image = summary["total_s"] * 1000 / len(subset)
            per_sec = len(subset) / summary["total_s"]
            print(f"{batch:>6} {per_image:>16.2f} {per_sec:>15.1f}")
            out[device][batch] = {
                "ms_per_image": round(per_image, 2),
                "images_per_s": round(per_sec, 1),
                "latency_ms_per_batch_p50": summary["latency_ms"]["p50"],
                "gpu_allocated_mb": summary["memory_mb"]["gpu_allocated"],
            }

    if len(devices) > 1:
        fast, slow = devices[0], devices[1]
        best = max(batches)
        speedup = (out[slow][best]["ms_per_image"] / out[fast][best]["ms_per_image"])
        out["speedup"] = {f"{fast}_vs_{slow}_at_batch_{best}": round(speedup, 2)}
        print(f"\nускорение {fast} против {slow} при батче {best}: {speedup:.2f}x")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="m2_image/data/imagenette_val")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--perf-devices", nargs="*", default=[],
                    help="замерить скорость на этих устройствах, например: mps cpu")
    ap.add_argument("--perf-batches", type=int, nargs="*", default=[1, 4, 16, 32],
                    help="размеры батча для замера выигрыша от батчинга")
    ap.add_argument("--perf-images", type=int, default=128,
                    help="сколько изображений гонять в замере скорости")
    args = ap.parse_args()

    from datasets import load_from_disk

    device = args.device or get_device()
    ds = load_from_disk(args.data)
    if args.limit and args.limit < len(ds):
        ds = ds.shuffle(seed=42).select(range(args.limit))

    label_names = ds.features["label"].names
    idx2cat = build_class_mapping(label_names)
    print(f"выборка: {len(ds)} | классов: {len(label_names)} | устройство: {device}")

    images = [r["image"].convert("RGB") for r in ds]
    y_true = [idx2cat[r["label"]] for r in ds]

    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["categories"]
    cat_names = list(catalog)

    # ---------------- zero-shot ------------------------------------------- #
    results = {}
    for with_dis in (False, True):
        tag = "с дистракторами" if with_dis else "только каталог"
        matcher = CategoryMatcher(args.model, device=device, with_distractors=with_dis)

        preds, scores, top3, bench = [], [], [], Bench(
            f"m2_clip_{'dis' if with_dis else 'nodis'}",
            device=device,
            extra={"model": args.model, "batch": args.batch},
        )
        with bench:
            for i in range(0, len(images), args.batch):
                chunk = images[i : i + args.batch]
                with bench.step():
                    out = matcher.classify(chunk, top_k=3)
                preds.extend(o["predicted"] for o in out)
                scores.extend(o["score"] for o in out)
                top3.extend([t["category"] for t in o["top_k"]] for o in out)

        top1 = float(np.mean([p == t for p, t in zip(preds, y_true)]))
        top3_acc = float(np.mean([t in k for k, t in zip(top3, y_true)]))
        print(f"\nzero-shot ({tag}): top-1 {top1:.3f} | top-3 {top3_acc:.3f}")

        results[f"clip_{'with' if with_dis else 'without'}_distractors"] = {
            "top1_accuracy": round(top1, 4),
            "top3_accuracy": round(top3_acc, 4),
            "performance": bench.summary(),
        }
        if with_dis:
            zs_preds, zs_scores = preds, scores

    # ---------------- обученный классификатор ------------------------------ #
    if not args.skip_baseline:
        # обученный классификатор сравнивается только на тех категориях,
        # которые есть в тестовом датасете
        eval_cats = [c for c in cat_names if catalog[c].get("eval_source_class")]
        src = [catalog[c]["eval_source_class"] for c in eval_cats]
        base = SupervisedBaseline(src, device)
        src2cat = dict(zip(src, eval_cats))

        bpreds, bbench = [], Bench("m2_resnet50", device=device, extra={"batch": args.batch})
        with bbench:
            for i in range(0, len(images), args.batch):
                with bbench.step():
                    bpreds.extend(base.predict(images[i : i + args.batch]))
        bpreds = [src2cat[p] for p in bpreds]
        bacc = float(np.mean([p == t for p, t in zip(bpreds, y_true)]))
        print(f"обученный ResNet50 (10 классов): top-1 {bacc:.3f}")
        results["resnet50_supervised"] = {
            "top1_accuracy": round(bacc, 4),
            "performance": bbench.summary(),
        }

    # ---------------- детекция несоответствия ------------------------------ #
    # Каждое изображение проверяется дважды: с верной категорией (честное
    # объявление) и с заведомо неверной (подмена). Решение принимается той же
    # функцией decide, что и в predict.py, — с порогом уверенности и корзиной
    # review, а не простым сравнением предсказания с заявкой.
    wrong = [cat_names[(cat_names.index(t) + 3) % len(cat_names)] for t in y_true]
    honest = [decide(p, sc, t) for p, sc, t in zip(zs_preds, zs_scores, y_true)]
    swapped = [decide(p, sc, w) for p, sc, w in zip(zs_preds, zs_scores, wrong)]
    n = len(y_true)

    def share(decisions, value):
        return round(sum(d == value for d in decisions) / n, 4)

    rej_honest, rej_swapped = honest.count("reject"), swapped.count("reject")
    mismatch = {
        "honest": {d: share(honest, d) for d in ("allow", "review", "reject")},
        "swapped": {d: share(swapped, d) for d in ("allow", "review", "reject")},
        # подмена поймана, если объявление не опубликовано автоматически
        "recall_mismatch": round(1 - swapped.count("allow") / n, 4),
        "recall_mismatch_reject": round(rej_swapped / n, 4),
        # при равном числе честных и подменённых объявлений
        "precision_reject": round(rej_swapped / (rej_swapped + rej_honest), 4)
        if rej_swapped + rej_honest else 0.0,
        "false_reject_rate": round(rej_honest / n, 4),
        "honest_to_review": share(honest, "review"),
        "note": "honest — честные объявления, swapped — подменённая категория; "
                "доли корзин считаются от числа изображений",
    }
    print(f"\nчестные:   {mismatch['honest']}")
    print(f"подмены:   {mismatch['swapped']}")
    print(f"recall подмены {mismatch['recall_mismatch']:.3f} | "
          f"precision(reject) {mismatch['precision_reject']:.3f} | "
          f"ложные отклонения {mismatch['false_reject_rate']:.3f}")

    perf = {}
    if args.perf_devices:
        perf = perf_sweep(images, args.model, args.perf_devices,
                          args.perf_batches, args.perf_images)

    metrics = {
        "dataset": args.data,
        "n_samples": len(ds),
        "device": device,
        "batch_size": args.batch,
        "match_threshold": MATCH_THRESHOLD,
        **results,
        "mismatch_detection": mismatch,
        "perf_sweep": perf,
    }
    out = ROOT / "report" / "metrics" / "m2_image.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nметрики →", out)


if __name__ == "__main__":
    main()
