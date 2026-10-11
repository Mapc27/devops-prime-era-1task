#!/usr/bin/env python3
"""M5 — сборка тестовых кейсов для проверки вердиктов.

Кейсы строятся систематически: перебираются сочетания решений модулей M1-M4,
включая случаи, где модуль вообще не отработал (пользователь не приложил видео
или голосовое). Эталонный вердикт вычисляется правилом, а не проставляется
руками — так в эталоне не будет опечаток.
"""

from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verdict import rule_verdict  # noqa: E402

# правдоподобные выходы модулей для каждого решения
FIXTURES = {
    "m1_text": {
        "allow": {"decision": "allow", "toxicity": 0.01, "reason": None},
        "review": {"decision": "review", "toxicity": 0.61,
                   "top_violation_ru": "оскорбление",
                   "reason": "пограничная формулировка в описании"},
        "reject": {"decision": "reject", "toxicity": 0.99,
                   "top_violation_ru": "нецензурная лексика",
                   "reason": "в описании нецензурная лексика"},
    },
    "m2_image": {
        "allow": {"decision": "allow", "match": True, "reason": None},
        "review": {"decision": "review", "match": False,
                   "reason": "модель не уверена в содержимом фотографии"},
        "reject": {"decision": "reject", "match": False,
                   "reason": "на фото распознано «Скриншот экрана», "
                             "а заявлена категория «Техника: ноутбуки»"},
    },
    "m3_audio": {
        "allow": {"decision": "allow", "reason": None},
        "review": {"decision": "review", "reason": "низкое качество записи"},
        "reject": {"decision": "reject", "top_violation_ru": "угроза",
                   "reason": "в голосовом сообщении угроза покупателю"},
    },
    "m4_video": {
        "allow": {"decision": "allow", "declared_present": True, "reason": None},
        "review": {"decision": "review", "declared_present": False,
                   "reason": "в ролике не найден объект заявленной категории"},
    },
}


def main() -> None:
    cases = []

    # базовый набор: всё чисто
    cases.append({"name": "всё чисто", "checks": {
        k: FIXTURES[k]["allow"] for k in FIXTURES}})

    # по одному нарушению за раз
    for module, decisions in FIXTURES.items():
        for decision in decisions:
            if decision == "allow":
                continue
            checks = {k: FIXTURES[k]["allow"] for k in FIXTURES}
            checks[module] = decisions[decision]
            cases.append({"name": f"{module}: {decision}", "checks": checks})

    # сочетания нарушений в двух модулях
    for m1d, m2d in product(["review", "reject"], ["review", "reject"]):
        checks = {k: FIXTURES[k]["allow"] for k in FIXTURES}
        checks["m1_text"] = FIXTURES["m1_text"][m1d]
        checks["m2_image"] = FIXTURES["m2_image"][m2d]
        cases.append({"name": f"текст:{m1d} + фото:{m2d}", "checks": checks})

    # неполные объявления: часть контента отсутствует
    cases.append({"name": "только текст, чисто", "checks": {
        "m1_text": FIXTURES["m1_text"]["allow"]}})
    cases.append({"name": "только текст, нарушение", "checks": {
        "m1_text": FIXTURES["m1_text"]["reject"]}})
    cases.append({"name": "текст чист, видео не подтвердило товар", "checks": {
        "m1_text": FIXTURES["m1_text"]["allow"],
        "m4_video": FIXTURES["m4_video"]["review"]}})
    cases.append({"name": "голосовое с угрозой, остальное чисто", "checks": {
        "m1_text": FIXTURES["m1_text"]["allow"],
        "m3_audio": FIXTURES["m3_audio"]["reject"]}})

    for c in cases:
        c["expected_verdict"] = rule_verdict(c["checks"])

    out = Path(__file__).parent / "data" / "cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

    dist: dict = {}
    for c in cases:
        dist[c["expected_verdict"]] = dist.get(c["expected_verdict"], 0) + 1
    print(f"кейсов: {len(cases)} | распределение вердиктов: {dist}")
    print("→", out)


if __name__ == "__main__":
    main()
