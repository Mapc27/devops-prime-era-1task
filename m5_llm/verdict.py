#!/usr/bin/env python3
"""M5 — итоговый вердикт и объяснение причины (локальная LLM).

Принципиальное разделение ответственности:

  * Вердикт считается ДЕТЕРМИНИРОВАННЫМ правилом (rule_verdict). Это надёжно,
    воспроизводимо и проверяемо, а решение о блокировке объявления — не то
    место, где уместна вероятностная генерация.

  * LLM отвечает за то, что правилом не выражается: связное объяснение
    на русском языке, понятное продавцу, и сводка для модератора.

LLM получает уже готовые числовые результаты M1-M4 и НЕ анализирует контент
заново. Совпадение её вердикта с правилом — метрика надёжности: расхождение
означает, что модель не удержала контекст.

Запуск (нужен работающий ollama serve):
    python m5_llm/verdict.py --case m5_llm/data/cases.json --index 0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:3b-instruct"  # замер показал: точнее 7B и вдвое быстрее, см. отчёт, раздел 5.5

# приоритет решений: побеждает самое строгое
SEVERITY = {"allow": 0, "review": 1, "reject": 2}

SYSTEM_PROMPT = """Ты — помощник модератора площадки частных объявлений.

Тебе передают ГОТОВЫЕ результаты автоматических проверок объявления.
Твоя задача — не анализировать контент заново, а:
1) назвать итоговый вердикт,
2) объяснить его продавцу вежливо, в одном-двух предложениях, без обвинений,
3) дать краткую техническую сводку для модератора.

Правило вердикта — строго: если хотя бы одна проверка дала reject, итог reject;
иначе если хотя бы одна дала review, итог review; иначе allow.

Отвечай ТОЛЬКО валидным JSON без markdown-разметки."""

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["allow", "review", "reject"]},
        "reason_for_user": {"type": "string"},
        "internal_note": {"type": "string"},
    },
    "required": ["verdict", "reason_for_user", "internal_note"],
}


def rule_verdict(checks: dict) -> str:
    """Детерминированный вердикт — эталон, с которым сверяется ответ модели."""
    decisions = [v.get("decision", "allow") for v in checks.values() if isinstance(v, dict)]
    if not decisions:
        return "allow"
    return max(decisions, key=lambda d: SEVERITY.get(d, 0))


def build_prompt(checks: dict) -> str:
    lines = ["Результаты проверок объявления:", ""]
    titles = {
        "m1_text": "Текст объявления",
        "m2_image": "Фотография",
        "m3_audio": "Голосовой комментарий",
        "m4_video": "Видеообзор",
    }
    for key, title in titles.items():
        res = checks.get(key)
        if not res:
            lines.append(f"- {title}: не предоставлен")
            continue
        detail = res.get("reason") or res.get("top_violation_ru") or "нарушений не выявлено"
        lines.append(f"- {title}: решение={res['decision']}, детали: {detail}")
    lines.append("")
    lines.append("Сформулируй итоговый вердикт и объяснение.")
    return "\n".join(lines)


class VerdictGenerator:
    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url

    def available(self) -> bool:
        """Установлена ли именно эта модель.

        Сравнение по полному имени: по префиксу «qwen2.5» установленная 7B
        выдавала себя за 3B, и запрос падал уже на генерации.
        """
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            return any(m["name"] == wanted for m in r.json()["models"])
        except Exception:
            return False

    def generate(self, checks: dict, temperature: float = 0.2,
                 enforce_rule: bool = False) -> dict:
        """Ответ модели по результатам проверок.

        enforce_rule — рабочий режим: если вердикт модели разошёлся с правилом,
        объяснение запрашивается повторно уже с готовым вердиктом. Иначе продавец
        прочитал бы «объявление отправлено на проверку» при фактическом отказе.
        Бенчмарк вызывает без него: там меряется именно первая попытка.
        """
        result = self._ask(checks, build_prompt(checks), temperature)
        result["retried"] = False
        if enforce_rule and not result["agrees_with_rule"]:
            prompt = (build_prompt(checks)
                      + f"\n\nИтоговый вердикт уже вычислен правилом: "
                        f"{result['rule_verdict']}. Используй именно его.")
            first_perf = result["perf"]
            result = self._ask(checks, prompt, temperature)
            result["retried"] = True
            result["first_attempt_perf"] = first_perf
        # решение всегда принимает правило, модель только формулирует текст
        result["final_verdict"] = result["rule_verdict"]
        return result

    def _ask(self, checks: dict, user_prompt: str, temperature: float) -> dict:
        payload = {
            "model": self.model,
            "stream": False,
            "format": JSON_SCHEMA,  # структурированный вывод на стороне сервера
            "options": {"temperature": temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        t0 = time.perf_counter()
        r = requests.post(f"{self.url}/api/chat", json=payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        elapsed = time.perf_counter() - t0

        raw = data["message"]["content"]
        try:
            parsed = json.loads(raw)
            valid_json = True
        except json.JSONDecodeError:
            parsed = {"verdict": None, "reason_for_user": raw, "internal_note": ""}
            valid_json = False

        expected = rule_verdict(checks)
        ns = 1e9
        return {
            "llm": parsed,
            "rule_verdict": expected,
            "valid_json": valid_json,
            "agrees_with_rule": parsed.get("verdict") == expected,
            "perf": {
                "total_s": round(elapsed, 2),
                "prompt_tokens": data.get("prompt_eval_count"),
                "generated_tokens": data.get("eval_count"),
                "generation_tps": round(
                    data["eval_count"] / (data["eval_duration"] / ns), 1
                ) if data.get("eval_duration") else None,
                "prompt_tps": round(
                    data["prompt_eval_count"] / (data["prompt_eval_duration"] / ns), 1
                ) if data.get("prompt_eval_duration") else None,
                "load_s": round(data.get("load_duration", 0) / ns, 2),
            },
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="m5_llm/data/cases.json")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    gen = VerdictGenerator(args.model)
    if not gen.available():
        raise SystemExit(
            f"модель {args.model} недоступна.\n"
            f"запустите: ollama serve  и  ollama pull {args.model}"
        )

    cases = json.loads(Path(args.case).read_text(encoding="utf-8"))
    case = cases[args.index]
    print(f"кейс: {case['name']}\n")
    print(build_prompt(case["checks"]))
    print("\n--- ответ модели ---")
    print(json.dumps(gen.generate(case["checks"], enforce_rule=True),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
