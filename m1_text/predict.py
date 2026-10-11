#!/usr/bin/env python3
"""M1 — детекция нарушений в тексте объявления.

Модель многоклассовая (multilabel): каждый вид нарушения оценивается независимо,
поэтому один текст может быть одновременно и оскорблением, и угрозой.
Гранулярность нужна модулю M5 — чтобы объяснить пользователю причину отказа.

Примеры:
    python m1_text/predict.py --text "Продам гараж, недорого"
    python m1_text/predict.py --file m1_text/data/samples.csv --column text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import Bench, get_device  # noqa: E402

# Решение принимает модель с лучшим покрытием видов нарушений: замер показал,
# что лёгкая multilabel-модель детектирует только прямые оскорбления и
# пропускает угрозы и презрение почти полностью (см. отчёт, раздел 5.1).
MODEL_ID = "s-nlp/russian_toxicity_classifier"

# Но она бинарная и не говорит, ЧТО именно нарушено, а модулю M5 это нужно
# для объяснения продавцу. Поэтому вид нарушения добавляет вторая, лёгкая
# модель — и только для текстов, которые уже помечены. На чистом потоке
# (95% объявлений) она не запускается вовсе.
EXPLAIN_MODEL_ID = "cointegrated/rubert-tiny-toxicity"

# человекочитаемые названия для объяснения пользователю
LABELS_RU = {
    "non-toxic": "нарушений не выявлено",
    "neutral": "нарушений не выявлено",
    "insult": "оскорбление",
    "obscenity": "нецензурная лексика",
    "threat": "угроза",
    "dangerous": "опасный контент",
    "toxic": "недопустимый контент",
}

# как называется класс «чистый текст» у разных моделей
CLEAN_LABELS = ("non-toxic", "neutral", "not_toxic", "ok")

# Минимальная уверенность, при которой вид нарушения называется вслух.
# Без этого порога argmax по видам нарушений всегда что-нибудь выдаёт: у лёгкой
# модели голова dangerous срабатывает широко и выигрывает даже на текстах,
# которые сама модель считает чистыми. Назвать продавцу выдуманную причину
# хуже, чем честно отправить объявление на ручную проверку.
MIN_EXPLAIN_SCORE = 0.5

THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"
DEFAULT_THRESHOLDS = {"review": 0.5, "reject": 0.8}


def load_thresholds() -> dict:
    """Пороги подбираются evaluate.py по валидационной выборке.

    До первого прогона оценки используются значения по умолчанию.
    """
    if THRESHOLDS_PATH.exists():
        return json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    return dict(DEFAULT_THRESHOLDS)


class TextModerator:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        explain_model_id: str | None = EXPLAIN_MODEL_ID,
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        # модель лёгкая, MPS не даёт выигрыша на коротких текстах — по умолчанию CPU
        self.device = device or "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model_id = model_id
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device).eval()
        cfg = self.model.config
        self.labels = [cfg.id2label[i] for i in range(cfg.num_labels)]

        # Модели токсичности бывают двух устройств, и путать их нельзя:
        #   multilabel — независимые сигмоиды, виды нарушений не исключают друг друга;
        #   binary     — один softmax на два класса.
        # Тип берётся из конфига модели, а не задаётся в коде, — иначе сравнить
        # две модели между собой было бы невозможно.
        self.multilabel = cfg.problem_type == "multi_label_classification" or (
            cfg.problem_type is None and cfg.num_labels > 2
        )
        self.clean_label = next(
            (lbl for lbl in self.labels if lbl.lower() in CLEAN_LABELS), None
        )
        if self.clean_label is None:
            raise SystemExit(
                f"не удалось определить класс «чистый текст» среди {self.labels}. "
                f"Добавьте его название в CLEAN_LABELS."
            )
        self.thresholds = load_thresholds()

        # вторая модель грузится только если она отличается от основной
        # и нужна для разбивки по видам нарушений
        self._explain_id = (
            explain_model_id if explain_model_id and explain_model_id != model_id else None
        )
        self._explainer: "TextModerator | None" = None

    @property
    def explainer(self) -> "TextModerator | None":
        """Ленивая загрузка модели-объяснителя: только при первом нарушении."""
        if self._explain_id and self._explainer is None:
            self._explainer = TextModerator(
                self._explain_id, device=self.device, explain_model_id=None
            )
        return self._explainer

    def scores(self, texts: list[str]) -> list[dict[str, float]]:
        """Вероятности по классам модели."""
        enc = self.tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True, max_length=512
        ).to(self.device)
        with self.torch.inference_mode():
            logits = self.model(**enc).logits
        activation = self.torch.sigmoid if self.multilabel else self.torch.softmax
        probs = (
            activation(logits) if self.multilabel else activation(logits, dim=-1)
        ).cpu().numpy()
        return [dict(zip(self.labels, row.tolist())) for row in probs]

    def toxicity(self, scores: dict[str, float]) -> float:
        """Единая оценка нарушения: вероятность того, что текст НЕ чистый.

        Считается одинаково для обоих устройств моделей, поэтому пороги
        и метрики сопоставимы между ними.
        """
        return 1.0 - scores.get(self.clean_label, 0.0)

    def decide(self, text: str, always_explain: bool = False) -> dict:
        scores = self.scores([text])[0]
        return self._decide_from(text, scores, always_explain)

    def _decide_from(self, text: str, scores: dict[str, float],
                     always_explain: bool = False) -> dict:
        tox = self.toxicity(scores)
        if tox >= self.thresholds["reject"]:
            decision = "reject"
        elif tox >= self.thresholds["review"]:
            decision = "review"
        else:
            decision = "allow"

        result = {
            "text": text[:200],
            "toxicity": round(tox, 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "decision": decision,
            "decided_by": self.model_id,
            # по каким порогам принято решение — без этого невозможно понять,
            # почему на одном и том же тексте вердикт разный
            "thresholds": {
                "review": self.thresholds["review"],
                "reject": self.thresholds["reject"],
                "fitted_on": self.thresholds.get("fitted_on", {}).get("dataset", "по умолчанию"),
            },
            "top_violation": None,
            "top_violation_ru": None,
        }
        # вид нарушения нужен только для помеченных текстов; на чистом потоке
        # вторая модель не запускается вовсе
        if decision == "allow" and not always_explain:
            return result

        # вид нарушения: своя разбивка, если модель многоклассовая,
        # иначе — у модели-объяснителя
        detail = scores
        if len(self.labels) <= 2 and self.explainer is not None:
            detail = self.explainer.scores([text])[0]
            result["explained_by"] = self._explain_id

        violations = {
            k: v
            for k, v in detail.items()
            if k.lower() not in CLEAN_LABELS
        }
        if violations:
            top = max(violations, key=violations.get)
            result["violation_scores"] = {k: round(v, 4) for k, v in detail.items()}
            if violations[top] >= MIN_EXPLAIN_SCORE:
                result["top_violation"] = top
                result["top_violation_ru"] = LABELS_RU.get(top, top)
            else:
                result["top_violation_ru"] = (
                    "вид нарушения не определён автоматически, нужна ручная проверка"
                )
        return result

    def decide_batch(self, texts: list[str], batch_size: int = 32,
                     always_explain: bool = False) -> list[dict]:
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            for text, sc in zip(chunk, self.scores(chunk)):
                out.append(self._decide_from(text, sc, always_explain))
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Модерация текста объявления")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="один текст")
    src.add_argument("--file", help="CSV-файл для пакетной обработки")
    ap.add_argument("--column", default="text", help="колонка с текстом в CSV")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--bench", action="store_true", help="сохранить замеры в report/")
    ap.add_argument("--explain", action="store_true",
                    help="запустить модель-объяснитель даже для чистых текстов — "
                         "показывает обе ступени каскада")
    ap.add_argument("--no-cascade", action="store_true",
                    help="только решающая модель, без второй ступени")
    args = ap.parse_args()

    mod = TextModerator(
        args.model, explain_model_id=None if args.no_cascade else EXPLAIN_MODEL_ID
    )

    if args.text:
        print(json.dumps(mod.decide(args.text, args.explain),
                         ensure_ascii=False, indent=2))
        return

    import pandas as pd

    df = pd.read_csv(args.file)
    texts = df[args.column].astype(str).tolist()

    with Bench("m1_text_predict", device=mod.device, extra={"model": args.model}) as b:
        results = []
        for text in texts:
            with b.step():
                results.append(mod.decide(text))

    counts = {d: 0 for d in ("allow", "review", "reject")}
    for r in results:
        counts[r["decision"]] += 1
    print(f"\nВсего: {len(results)} | " + " | ".join(f"{k}: {v}" for k, v in counts.items()))
    b.report()
    if args.bench:
        print("замеры →", b.save())


if __name__ == "__main__":
    main()
