#!/usr/bin/env python3
"""M4 — сборка тестового набора видеообзоров.

Публичных размеченных роликов «видеообзор товара» нет, поэтому набор
собирается из фотографий: камера медленно наезжает и панорамирует по кадру.
Для детектора это близко к тому, что снимает продавец, обходя товар, —
объект остаётся в кадре, меняются масштаб и положение.

Ограничение, которое надо держать в голове при чтении метрик: в таком ролике
нет смазов, смены освещения и ухода объекта из кадра. Метрики получаются
оптимистичными. Живые записи с телефона кладутся в data/raw/ и подхватываются
тем же манифестом.

Исходные кадры — первые четыре фотографии трёх классов Imagenette:
    python m2_image/download_data.py
    python m4_video/prepare_frames.py

Запуск:
    python m4_video/make_testset.py
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

DATA = Path(__file__).parent / "data"
FRAMES = DATA / "frames"
VIDEO = DATA / "video"

DURATION = 6  # секунд
FPS = 25

# какая категория площадки соответствует исходной фотографии
SOURCE_TO_CATEGORY = {
    "English_springer": "Животные: собаки",
    "garbage_truck": "Спецтехника",
    "golf_ball": "Спорт: мячи",
}


def ken_burns(image: Path, out: Path) -> None:
    """Медленный наезд с панорамированием."""
    frames = DURATION * FPS
    vf = (
        "scale=1280:-2,"
        f"zoompan=z='min(zoom+0.0015,1.4)':d={frames}"
        ":x='iw/2-(iw/zoom/2)+sin(on/40)*80'"
        ":y='ih/2-(ih/zoom/2)':s=1280x720:fps=" + str(FPS) + ","
        "format=yuv420p"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loop", "1", "-i", str(image), "-vf", vf,
         "-t", str(DURATION), "-c:v", "libx264", "-preset", "veryfast", str(out)],
        check=True, capture_output=True,
    )


def main() -> None:
    VIDEO.mkdir(parents=True, exist_ok=True)
    raw = json.loads(
        (Path(__file__).resolve().parent.parent / "catalog.json").read_text(encoding="utf-8")
    )["categories"]
    # подменную категорию берём только из тех, что видеомодуль умеет проверять
    categories = [n for n, c in raw.items() if c["coco_classes"]]

    rows = []
    images = sorted(FRAMES.glob("*.jpg"))
    if not images:
        raise SystemExit(f"нет исходных изображений в {FRAMES}")

    for img in images:
        source = "_".join(img.stem.split("_")[:-1])
        true_cat = SOURCE_TO_CATEGORY.get(source)
        if true_cat is None:
            print(f"пропуск {img.name}: нет категории для «{source}»")
            continue

        out = VIDEO / f"{img.stem}.mp4"
        ken_burns(img, out)
        print(f"  {out.name}")

        # верная заявка: категория совпадает с содержимым
        rows.append({"video": out.name, "declared": true_cat,
                     "true_category": true_cat, "should_pass": 1})
        # подмена: заявлена другая категория, система обязана не подтвердить
        wrong = categories[(categories.index(true_cat) + 4) % len(categories)]
        rows.append({"video": out.name, "declared": wrong,
                     "true_category": true_cat, "should_pass": 0})

    manifest = DATA / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "declared", "true_category", "should_pass"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nроликов: {len(set(r['video'] for r in rows))} | "
          f"проверок: {len(rows)} | манифест → {manifest}")


if __name__ == "__main__":
    main()
