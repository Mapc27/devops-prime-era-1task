#!/usr/bin/env python3
"""M1 — замер HTTP-сервиса так, как его видит клиент.

Латентность модели и латентность сервиса — разные числа: ко второй добавляются
разбор HTTP, валидация pydantic и сериализация JSON. Замер идёт через
TestClient FastAPI, то есть через полный стек приложения без сети.

Запуск:
    python m1_text/bench_api.py --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.bench import Bench  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(Path(__file__).parent / "data" / "samples.csv"))
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    import pandas as pd
    from fastapi.testclient import TestClient

    from api import app

    texts = pd.read_csv(args.dataset)["text"].astype(str).tolist()
    texts = (texts * (args.n // len(texts) + 1))[: args.n]

    t0 = time.perf_counter()
    with TestClient(app) as client:  # lifespan: загрузка и прогрев моделей
        startup_s = time.perf_counter() - t0

        bench = Bench("m1_api", device="cpu", extra={"mode": "HTTP POST /predict, TestClient"})
        decisions: dict[str, int] = {}
        with bench:
            for text in texts:
                with bench.step():
                    r = client.post("/predict", json={"text": text})
                r.raise_for_status()
                d = r.json()["decision"]
                decisions[d] = decisions.get(d, 0) + 1

    summary = bench.summary()
    summary["startup_s"] = round(startup_s, 2)
    summary["decisions"] = decisions
    bench.report()
    print(f"  старт сервиса с прогревом: {summary['startup_s']} с | решения: {decisions}")

    out = ROOT / "report" / "metrics" / "m1_api.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("метрики →", out)


if __name__ == "__main__":
    main()
