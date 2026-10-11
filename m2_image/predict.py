#!/usr/bin/env python3
"""M2 — проверка соответствия фотографии заявленной категории.

Zero-shot: модель сопоставляет изображение с ТЕКСТОВЫМ описанием категории,
а не с фиксированным набором классов. Добавление новой категории на площадке
не требует обучения — достаточно дописать запись в общий catalog.json.

Две оптимизации, существенные для продакшена:
  1. Эмбеддинги текстов категорий считаются ОДИН раз при инициализации.
     На запрос приходится только кодирование изображения.
  2. Промпт-ансамблирование: на категорию несколько формулировок, их эмбеддинги
     усредняются. Снижает чувствительность к неудачной формулировке.

Примеры:
    python m2_image/predict.py --image photo.jpg
    python m2_image/predict.py --image photo.jpg --category "Инструменты: бензопилы"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import Bench, get_device  # noqa: E402

MODEL_ID = "openai/clip-vit-base-patch32"
CATALOG_PATH = Path(__file__).resolve().parent.parent / "catalog.json"

# порог уверенности: ниже него решение не принимается автоматически
MATCH_THRESHOLD = 0.5


def decide(predicted: str, score: float, declared: str) -> str:
    """Решение по заявленной категории — одно для инференса и для оценки.

    allow  — распознана заявленная категория с уверенностью не ниже порога;
    review — модель не уверена ни в какой категории;
    reject — модель уверенно видит на фото другую категорию.
    """
    if score < MATCH_THRESHOLD:
        return "review"
    return "allow" if predicted == declared else "reject"


class CategoryMatcher:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        with_distractors: bool = True,
    ):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device or get_device()
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        self.catalog = dict(catalog["categories"])
        if with_distractors:
            self.catalog.update(catalog["distractors"])
        self.names = list(self.catalog)
        self.text_features = self._encode_categories()

    def _encode_categories(self):
        """Эмбеддинг категории = усреднённый эмбеддинг её промптов."""
        vectors = []
        for name in self.names:
            prompts = [f"a photo of {p}." for p in self.catalog[name]["clip_prompts"]]
            enc = self.processor(text=prompts, return_tensors="pt", padding=True)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self.torch.inference_mode():
                # transformers 5.x возвращает объект; pooler_output — это уже
                # спроецированный в общее пространство эмбеддинг (512-d)
                feats = self.model.get_text_features(**enc).pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)
            mean = feats.mean(dim=0)
            vectors.append(mean / mean.norm())
        return self.torch.stack(vectors)

    def _image_features(self, images: list):
        enc = self.processor(images=images, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with self.torch.inference_mode():
            feats = self.model.get_image_features(**enc).pooler_output
        return feats / feats.norm(dim=-1, keepdim=True)

    def classify(self, images: list, top_k: int = 3) -> list[dict]:
        """Вероятности по категориям для пачки изображений."""
        img_feats = self._image_features(images)
        # температура обучена вместе с моделью, берём её же
        logits = self.model.logit_scale.exp() * img_feats @ self.text_features.T
        probs = logits.softmax(dim=-1).cpu()

        out = []
        for row in probs:
            values, indices = row.topk(min(top_k, len(self.names)))
            out.append(
                {
                    "predicted": self.names[indices[0]],
                    "score": round(values[0].item(), 4),
                    "top_k": [
                        {"category": self.names[i], "score": round(v.item(), 4)}
                        for v, i in zip(values, indices)
                    ],
                }
            )
        return out

    def check(self, image, declared: str | None = None, top_k: int = 3) -> dict:
        result = self.classify([image], top_k)[0]
        if declared is None:
            return result

        declared_score = next(
            (t["score"] for t in result["top_k"] if t["category"] == declared), None
        )
        decision = decide(result["predicted"], result["score"], declared)
        match = decision == "allow"

        result["declared"] = declared
        result["declared_score"] = declared_score
        result["match"] = match
        result["decision"] = decision
        result["reason"] = (
            None
            if match
            else f"на фото распознано «{result['predicted']}», "
            f"а заявлена категория «{declared}»"
        )
        return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Проверка фото на соответствие категории")
    ap.add_argument("--image", required=True, help="путь к изображению")
    ap.add_argument("--category", help="заявленная пользователем категория")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default=None, help="cpu / mps (по умолчанию авто)")
    ap.add_argument("--no-distractors", action="store_true")
    ap.add_argument("--list-categories", action="store_true")
    args = ap.parse_args()

    if args.list_categories:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        print("\n".join(catalog["categories"]))
        return

    from PIL import Image

    matcher = CategoryMatcher(
        args.model, device=args.device, with_distractors=not args.no_distractors
    )
    image = Image.open(args.image).convert("RGB")

    with Bench("m2_image_predict", device=matcher.device, extra={"model": args.model}) as b:
        with b.step():
            result = matcher.check(image, args.category)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    b.report()


if __name__ == "__main__":
    main()
