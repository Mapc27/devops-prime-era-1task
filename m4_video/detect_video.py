#!/usr/bin/env python3
"""M4 — проверка видеообзора товара.

Задача: подтвердить, что в ролике действительно присутствует заявленный товар.

Два решения, определяющие работу модуля:

  1. Анализируется не каждый кадр, а каждый N-й. Соседние кадры почти одинаковы,
     поэтому полный проход тратит время впустую. Влияние N на качество и скорость
     измеряется в evaluate.py — параметр выбирается по графику, а не на глаз.

  2. Объект считается подтверждённым, только если он найден минимум в K кадрах.
     Одиночное срабатывание на одном кадре — почти наверняка ошибка детектора.

Примеры:
    python m4_video/detect_video.py --video обзор.mp4 --category "Велосипеды"
    python m4_video/detect_video.py --video обзор.mp4 --stride 10 --save
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import get_device, gpu_memory_mb, rss_mb  # noqa: E402

MODEL_ID = "yolo11n.pt"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"

STRIDE = 5  # анализировать каждый 5-й кадр
CONF = 0.35  # порог уверенности детектора
MIN_FRAMES = 3  # минимум кадров для подтверждения объекта


class VideoInspector:
    def __init__(self, model_id: str = MODEL_ID, device: str | None = None):
        from ultralytics import YOLO

        self.device = device or get_device()
        self.model = YOLO(model_id)
        self.model_id = model_id
        # из общего справочника берём только категории, которые вообще можно
        # подтвердить детектором: у остальных нет подходящего класса COCO
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["categories"]
        self.catalog = {n: c["coco_classes"] for n, c in raw.items() if c["coco_classes"]}

    def inspect(
        self,
        video: str | Path,
        declared: str | None = None,
        stride: int = STRIDE,
        conf: float = CONF,
        min_frames: int = MIN_FRAMES,
        save: bool = False,
    ) -> dict:
        t0 = time.perf_counter()
        rss0 = rss_mb()

        results = self.model.predict(
            source=str(video),
            vid_stride=stride,  # пропуск кадров выполняет сам загрузчик
            conf=conf,
            device=self.device,
            stream=True,  # генератор: видео не грузится в память целиком
            save=save,
            project=str(Path(__file__).parent / "out"),
            name=Path(video).stem,
            exist_ok=True,
            verbose=False,
        )

        frames_with_class: Counter = Counter()
        best_conf: dict[str, float] = {}
        n_frames = 0

        for r in results:
            n_frames += 1
            names = r.names
            seen = set()
            for box in r.boxes:
                cls = names[int(box.cls)]
                score = float(box.conf)
                seen.add(cls)
                best_conf[cls] = max(best_conf.get(cls, 0.0), score)
            frames_with_class.update(seen)

        elapsed = time.perf_counter() - t0

        confirmed = {
            cls: {
                "frames": cnt,
                "share_of_frames": round(cnt / n_frames, 3) if n_frames else 0,
                "best_confidence": round(best_conf[cls], 3),
            }
            for cls, cnt in frames_with_class.most_common()
            if cnt >= min_frames
        }

        out = {
            "video": Path(video).name,
            "frames_analyzed": n_frames,
            "elapsed_s": round(elapsed, 2),
            "fps": round(n_frames / elapsed, 1) if elapsed else 0,
            "device": self.device,
            "stride": stride,
            "objects": confirmed,
            "rejected_as_noise": {
                cls: cnt for cls, cnt in frames_with_class.items() if cnt < min_frames
            },
            "memory_mb": {
                "rss_delta": round(rss_mb() - rss0, 1),
                "gpu_allocated": round(gpu_memory_mb(), 1),
            },
        }

        if declared is not None:
            expected = self.catalog.get(declared)
            if expected is None:
                raise SystemExit(
                    f"категория «{declared}» не найдена.\n"
                    f"доступные: {', '.join(self.catalog)}"
                )
            found = [c for c in expected if c in confirmed]
            out["declared"] = declared
            out["expected_classes"] = expected
            out["found_classes"] = found
            out["declared_present"] = bool(found)
            out["decision"] = "allow" if found else "review"
            out["reason"] = (
                None
                if found
                else f"в ролике не найден объект категории «{declared}»; "
                f"обнаружено: {', '.join(confirmed) or 'ничего'}"
            )
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Проверка видеообзора товара")
    ap.add_argument("--video", help="путь к видеофайлу")
    ap.add_argument("--category", help="заявленная категория")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument("--save", action="store_true", help="сохранить видео с рамками")
    ap.add_argument("--list-categories", action="store_true")
    args = ap.parse_args()

    if not args.list_categories and not args.video:
        ap.error("укажите --video или --list-categories")

    if args.list_categories:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))["categories"]
        for name, cfg in raw.items():
            classes = cfg["coco_classes"]
            mark = ", ".join(classes) if classes else "— нет класса COCO, не проверяется"
            print(f"{name:30s} <- {mark}")
        return

    insp = VideoInspector(args.model, args.device)
    result = insp.inspect(
        args.video, args.category, args.stride, args.conf, args.min_frames, args.save
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
