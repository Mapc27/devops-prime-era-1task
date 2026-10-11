#!/usr/bin/env python3
"""Схема модулей для отчёта. Генерируется кодом, чтобы не расходиться с реализацией.

Измеренные показатели в правой колонке читаются из report/metrics/*.json.
Холст небольшой и шрифт крупный: в отчёте рисунок сжимается до ширины
полосы набора (~6.3 дюйма), и на большом холсте подписи становились
нечитаемыми.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).parent / "img" / "architecture.png"


def load(name):
    return json.loads((ROOT / "report" / "metrics" / name).read_text(encoding="utf-8"))


def n(x, d=2):
    return f"{x:.{d}f}".replace(".", ",")


m1 = load("m1_russian_toxicity_classifier_domain_final.json")
m1_lat = load("m1_single_cascade.json")["latency_ms"]
m2 = load("m2_image.json")
m3 = {r["model"]: r for r in load("m3_audio.json")}["whisper-small"]["conditions"]["clean"]
m4 = {r["stride"]: r for r in load("m4_video.json")["models"]["yolo11n.pt"]["devices"]["mps"]}[5]
m5 = {r["model"]: r for r in load("m5_llm.json")}["qwen2.5:3b-instruct"]

MODULES = [
    ("M1", "Текст", "описание", "Transformers",
     "russian_toxicity 178M\n+ rubert-tiny 11.8M",
     f"ROC-AUC {n(m1['roc_auc'], 3)}\np95 {m1_lat['p95']:.0f} мс"),
    ("M2", "Фото", "фотографии", "Transformers",
     "CLIP ViT-B/32\n151M, zero-shot",
     f"top-1 {n(m2['clip_with_distractors']['top1_accuracy'], 3)}\n"
     f"{n(m2['perf_sweep']['mps']['32']['ms_per_image'], 1)} мс/фото"),
    ("M3", "Аудио", "голосовое", "faster-whisper",
     "whisper-small\n244M, int8",
     f"WER {n(m3['wer'], 3)}\nRTF {n(m3['rtf_mean'])}"),
    ("M4", "Видео", "видеообзор", "Ultralytics",
     "yolo11n\n2.6M, COCO-80",
     f"accuracy {n(m4['accuracy'], 3)}\n{m4['frames_analyzed'] / m4['elapsed_s']:.0f} кадров/с"),
]

C_IN, C_MOD, C_LLM, C_OUT = "#E8EEF7", "#FFFFFF", "#F3E8F7", "#E8F7EE"
EDGE = "#4A5568"


def box(ax, x, y, w, h, text, face, size=11, weight="normal", edge=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                                facecolor=face, edgecolor=edge, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=size, weight=weight, linespacing=1.45)


def arrow(ax, x1, y1, x2, y2, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=1.0,
                                 color=EDGE, shrinkA=1, shrinkB=1))


fig, ax = plt.subplots(figsize=(9.5, 8.6))
ax.set_xlim(-2, 101); ax.set_ylim(-4, 100); ax.axis("off")

ax.text(50, 97, "Пять модулей на готовых моделях",
        ha="center", fontsize=14, weight="bold")

# колонки: вход -> библиотека -> модель -> метрика
col_x = [0, 20, 41, 70]
col_w = [16, 17, 25, 20]
ax.text(col_x[0] + col_w[0] / 2, 88, "Контент объявления", ha="center", fontsize=11, weight="bold")
ax.text(col_x[1] + col_w[1] / 2, 88, "Библиотека", ha="center", fontsize=11, weight="bold")
ax.text(col_x[2] + col_w[2] / 2, 88, "Готовая модель", ha="center", fontsize=11, weight="bold")
ax.text(col_x[3] + col_w[3] / 2, 88, "Измерено", ha="center", fontsize=11, weight="bold")

row_h, gap = 12.0, 2.8
top = 80
for i, (code, kind, inp, lib, model, metric) in enumerate(MODULES):
    y = top - i * (row_h + gap) - row_h
    box(ax, col_x[0], y, col_w[0], row_h, f"{code} · {kind}\n{inp}", C_IN, 11, "bold")
    box(ax, col_x[1], y, col_w[1], row_h, lib, C_MOD, 11)
    box(ax, col_x[2], y, col_w[2], row_h, model, C_MOD, 10.5)
    box(ax, col_x[3], y, col_w[3], row_h, metric, C_OUT, 11)
    for a in range(3):
        arrow(ax, col_x[a] + col_w[a], y + row_h / 2, col_x[a + 1], y + row_h / 2)

# результаты всех четырёх модулей собираются вертикальной шиной
bus_x = col_x[3] + col_w[3] + 4.5
bus_top = top - row_h / 2
bus_bottom = top - 3 * (row_h + gap) - row_h / 2
ax.plot([bus_x, bus_x], [bus_bottom, bus_top], color=EDGE, linewidth=1.0)
for i in range(4):
    y = top - i * (row_h + gap) - row_h / 2
    arrow(ax, col_x[3] + col_w[3], y, bus_x, y)

# нижний поток: правило -> LLM -> вердикт
bot_y, bot_h = 4, 12
box(ax, 0, bot_y, 28, bot_h,
    "Правило вердикта\n(код): самое строгое\nиз решений M1–M4", "#FFF6E5", 10.5, "bold")
box(ax, 34, bot_y, 32, bot_h,
    f"M5 · LLM локально\nqwen2.5-3b Q4, Ollama\n"
    f"{m5['generation_tps']['mean']:.0f} ток/с, {m5['memory']['model_mb'] / 1024:.1f} ГБ",
    C_LLM, 10.5)
box(ax, 72, bot_y, 27, bot_h,
    "Вердикт\nallow / review / reject\n+ объяснение", C_OUT, 10.5, "bold")

# ортогональная разводка: вниз по правому краю, затем влево к правилу
lane_y = bot_y + bot_h + 3.5
ax.plot([bus_x, bus_x], [lane_y, bus_bottom], color=EDGE, linewidth=1.0)
ax.plot([14, bus_x], [lane_y, lane_y], color=EDGE, linewidth=1.0)
arrow(ax, 14, lane_y, 14, bot_y + bot_h)
ax.text(bus_x - 3, lane_y + 1.3, "результаты проверок M1–M4",
        fontsize=10, ha="right", style="italic")

arrow(ax, 28.5, bot_y + bot_h / 2, 34, bot_y + bot_h / 2)
arrow(ax, 66.5, bot_y + bot_h / 2, 72, bot_y + bot_h / 2)

ax.text(50, -1.5,
        "Вердикт считает правило, LLM только формулирует объяснение",
        ha="center", fontsize=10, style="italic", color="#555")

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("схема →", OUT)
