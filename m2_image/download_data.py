#!/usr/bin/env python3
"""M2 — загрузка тестового набора изображений.

Imagenette — подмножество ImageNet из 10 легко различимых классов. Исходное
хранилище fast.ai из сети недоступно, поэтому берётся зеркало на Hugging Face
с тем же валидационным сплитом (3925 изображений). evaluate.py случайно
отбирает из него 500 с фиксированным seed.

Этот же набор — источник кадров для тестовых роликов M4
(см. m4_video/prepare_frames.py).

Запуск:
    python m2_image/download_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATA = Path(__file__).parent / "data"
DEFAULT = "Multimodal-Fatima/Imagenette_validation"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--out", default=str(DATA / "imagenette_val"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))

    print(f"источник: {args.dataset} [{args.split}]")
    print(f"изображений: {len(ds)} | классы: {ds.features['label'].names}")
    print("→", out)


if __name__ == "__main__":
    main()
