#!/usr/bin/env python3
"""M5 — измерение качества и производительности локальной LLM.

Что меряется:
  * доля синтаксически валидного JSON — если модель ломает формат,
    сервис не сможет разобрать ответ, и никакое качество текста не спасёт;
  * совпадение вердикта модели с детерминированным правилом;
  * скорость генерации и обработки промпта (токенов/с), время до ответа;
  * пиковая память процесса ollama.

Сравнение нескольких моделей запускается одной командой — это и есть
обоснование выбора модели для отчёта.

Запуск:
    python m5_llm/benchmark.py --models qwen2.5:7b-instruct llama3.1:8b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import psutil
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from verdict import MODEL, OLLAMA_URL, VerdictGenerator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

MIXED_WORD = re.compile(r"\b(?=\w*[а-яё])(?=\w*[a-z])\w+\b", re.IGNORECASE)
INTERNAL_TERMS = re.compile(r"\b(allow|review|reject|m[1-5]_\w+|decision)\b", re.IGNORECASE)
MAX_USER_TEXT = 400  # длиннее — уже не «одно-два предложения»


def text_issues(text: str) -> list[str]:
    """Формальные дефекты объяснения для продавца.

    Качество текста целиком автоматически не оценить, но самые частые дефекты
    малых моделей ловятся правилами: слово, склеенное из латиницы и кириллицы
    («кategoriей»), служебные термины системы в тексте для продавца, пустой
    или слишком длинный ответ.
    """
    issues = []
    if not text or not text.strip():
        return ["пустой текст"]
    if MIXED_WORD.search(text):
        issues.append("смешение алфавитов")
    if INTERNAL_TERMS.search(text):
        issues.append("служебные термины")
    if len(text) > MAX_USER_TEXT:
        issues.append("слишком длинный")
    return issues


def unload_all(url: str = OLLAMA_URL) -> None:
    """Выгрузить все загруженные модели.

    Ollama держит модель в памяти ещё пять минут после запроса. Если этого
    не сделать, при замере второй модели в памяти окажутся обе, и цифра
    получится суммой — именно так первый замер показал 7 ГБ для модели,
    которая весит 1.9 ГБ.
    """
    try:
        loaded = requests.get(f"{url}/api/ps", timeout=5).json().get("models", [])
        for m in loaded:
            requests.post(f"{url}/api/generate",
                          json={"model": m["name"], "keep_alive": 0}, timeout=30)
    except Exception:
        pass


def model_memory_mb(model: str, url: str = OLLAMA_URL) -> dict:
    """Память под конкретную модель.

    Берётся из /api/ps самого сервера, а не из RSS процессов: там указано,
    сколько занимает именно эта модель, без примеси соседних и без
    разделяемых страниц.
    """
    out = {"model_mb": None, "vram_mb": None, "loaded_models": 0}
    try:
        loaded = requests.get(f"{url}/api/ps", timeout=5).json().get("models", [])
        out["loaded_models"] = len(loaded)
        for m in loaded:
            if m["name"] == model:
                out["model_mb"] = round(m.get("size", 0) / 2**20, 1)
                out["vram_mb"] = round(m.get("size_vram", 0) / 2**20, 1)
    except Exception:
        pass

    # RSS процесса-рантайма для сверки с тем, что сообщает сервер
    runner = 0.0
    for proc in psutil.process_iter(["name", "cmdline", "memory_info"]):
        try:
            cmdline = " ".join(proc.info["cmdline"] or []).lower()
            if "llama-server" in (proc.info["name"] or "").lower() or "llama-server" in cmdline:
                runner += proc.info["memory_info"].rss / 2**20
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    out["runner_rss_mb"] = round(runner, 1)
    return out


def bench_model(model: str, cases: list[dict], verbose: bool = False,
                repeats: int = 1) -> dict:
    unload_all()
    gen = VerdictGenerator(model)
    if not gen.available():
        print(f"  модель {model} не найдена, пропуск (ollama pull {model})")
        return {}

    valid, agree, gen_tps, prompt_tps, totals = 0, 0, [], [], []
    mismatches, defects = [], []

    runs = [c for _ in range(repeats) for c in cases]
    for case in runs:
        r = gen.generate(case["checks"])
        valid += r["valid_json"]
        agree += r["agrees_with_rule"]
        totals.append(r["perf"]["total_s"])
        if r["perf"]["generation_tps"]:
            gen_tps.append(r["perf"]["generation_tps"])
        if r["perf"]["prompt_tps"]:
            prompt_tps.append(r["perf"]["prompt_tps"])

        issues = text_issues(r["llm"].get("reason_for_user", ""))
        if issues:
            defects.append({"case": case["name"], "issues": issues,
                            "text": r["llm"].get("reason_for_user", "")})
        if not r["agrees_with_rule"]:
            mismatches.append({
                "case": case["name"],
                "expected": r["rule_verdict"],
                "got": r["llm"].get("verdict"),
            })
        if verbose:
            mark = "OK " if r["agrees_with_rule"] else "РАСХОЖДЕНИЕ"
            print(f"  {mark} {case['name']:42s} "
                  f"{r['perf']['generation_tps']} ток/с")

    n = len(runs)
    ordered = sorted(totals)
    by_issue: dict[str, int] = {}
    for d in defects:
        for issue in d["issues"]:
            by_issue[issue] = by_issue.get(issue, 0) + 1
    return {
        "model": model,
        "n_cases": n,
        "repeats": repeats,
        "valid_json_rate": round(valid / n, 4),
        "verdict_agreement": round(agree / n, 4),
        "mismatches": mismatches,
        # доля ответов без формальных дефектов в тексте для продавца
        "clean_text_rate": round(1 - len(defects) / n, 4),
        "text_issues": by_issue,
        "text_defects": defects,
        "generation_tps": {
            "mean": round(statistics.fmean(gen_tps), 1) if gen_tps else None,
            "min": round(min(gen_tps), 1) if gen_tps else None,
            "max": round(max(gen_tps), 1) if gen_tps else None,
        },
        "prompt_tps_mean": round(statistics.fmean(prompt_tps), 1) if prompt_tps else None,
        "response_s": {
            "mean": round(statistics.fmean(totals), 2),
            # тот же способ, что в common/bench.py
            "p95": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 2),
        },
        "memory": model_memory_mb(model),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[MODEL])
    ap.add_argument("--cases", default="m5_llm/data/cases.json")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--repeats", type=int, default=1,
                    help="прогнать набор несколько раз — проверка воспроизводимости")
    args = ap.parse_args()

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    print(f"кейсов: {len(cases)}\n")

    results = []
    for model in args.models:
        print(f"=== {model} ===")
        r = bench_model(model, cases, args.verbose, args.repeats)
        if not r:
            continue
        results.append(r)
        print(f"  валидный JSON:      {r['valid_json_rate']:.1%}")
        print(f"  совпадение вердикта: {r['verdict_agreement']:.1%}")
        print(f"  текст без дефектов: {r['clean_text_rate']:.1%} {r['text_issues']}")
        print(f"  генерация:          {r['generation_tps']['mean']} ток/с")
        print(f"  обработка промпта:  {r['prompt_tps_mean']} ток/с")
        print(f"  ответ:              {r['response_s']['mean']} с (p95 {r['response_s']['p95']})")
        mem = r["memory"]
        print(f"  память модели:      {mem['model_mb']} МБ "
              f"(процесс {mem['runner_rss_mb']} МБ, загружено моделей: "
              f"{mem['loaded_models']})\n")

    out = ROOT / "report" / "metrics" / "m5_llm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("метрики →", out)


if __name__ == "__main__":
    main()
