#!/usr/bin/env python3
"""HTTP-обёртка над M1 — задел на следующий этап (разработка API).

Модель загружается один раз при старте приложения, а не на каждый запрос.

Запуск:
    .venv/bin/uvicorn m1_text.api:app --port 8000
"""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from predict import MODEL_ID, TextModerator  # noqa: E402

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.perf_counter()
    model = TextModerator()
    # Модель-объяснитель в CLI грузится лениво, но в сервисе это означало бы
    # 1.6 с на первом же запросе с нарушением — в 16 раз больше SLA.
    # Поэтому здесь обе модели загружаются и прогреваются при старте.
    model.decide("прогрев", always_explain=True)
    state["model"] = model
    state["startup_s"] = round(time.perf_counter() - t0, 2)
    print(f"модель загружена за {state['startup_s']} с")
    yield
    state.clear()


app = FastAPI(
    title="M1 — модерация текста объявлений",
    description="Детекция оскорблений, нецензурной лексики и угроз в тексте объявления",
    version="1.0.0",
    lifespan=lifespan,
)


class ModerateRequest(BaseModel):
    text: str = Field(..., max_length=5000, examples=["Продам гараж в центре, недорого"])


class BatchRequest(BaseModel):
    texts: list[str] = Field(..., max_length=100)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "startup_s": state.get("startup_s"),
    }


@app.post("/predict")
def predict(req: ModerateRequest) -> dict:
    t0 = time.perf_counter()
    result = state["model"].decide(req.text)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/predict/batch")
def predict_batch(req: BatchRequest) -> dict:
    t0 = time.perf_counter()
    results = state["model"].decide_batch(req.texts)
    return {
        "results": results,
        "count": len(results),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
