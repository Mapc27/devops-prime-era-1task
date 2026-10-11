#!/usr/bin/env python3
"""M4 — исходные кадры для тестовых роликов.

Ролики собираются из фотографий (см. make_testset.py). Фотографии берутся
из того же набора Imagenette, что и в M2: первые четыре изображения каждого
из трёх классов, которые детектор COCO в принципе может подтвердить.

Нужен скачанный набор M2:
    python m2_image/download_data.py
    python m4_video/prepare_frames.py
    python m4_video/make_testset.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRAMES = Path(__file__).parent / "data" / "frames"

# класс Imagenette -> префикс имени файла (его разбирает make_testset.py)
CLASSES = {
    "English springer": "English_springer",
    "garbage truck": "garbage_truck",
    "golf ball": "golf_ball",
}
PER_CLASS = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "m2_image" / "data" / "imagenette_val"))
    ap.add_argument("--out", default=str(FRAMES))
    args = ap.parse_args()
    out_dir = Path(args.out)

    from datasets import load_from_disk

    ds = load_from_disk(args.data)
    names = ds.features["label"].names
    wanted = {names.index(c): prefix for c, prefix in CLASSES.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    taken: dict[int, int] = defaultdict(int)
    for i, label in enumerate(ds["label"]):
        if label not in wanted or taken[label] >= PER_CLASS:
            continue
        taken[label] += 1
        out = out_dir / f"{wanted[label]}_{taken[label]}.jpg"
        ds[i]["image"].convert("RGB").save(out, quality=95)
        print(f"  {out.name}  <- индекс {i}")
        if all(taken[k] >= PER_CLASS for k in wanted):
            break

    print(f"кадров: {sum(taken.values())} → {out_dir}")


if __name__ == "__main__":
    main()
