#!/usr/bin/env python3
"""Сборка отчёта в .docx из измеренных метрик.

Все измеренные значения — таблицы и числа в тексте — читаются из
report/metrics/*.json, а не вписываются руками: после перезапуска оценки
достаточно пересобрать отчёт. В коде остаются только паспортные данные
моделей и датасетов (число параметров, размер корпуса).

Скриншоты кладутся в report/img/screens/ под именами из SCREENSHOTS; пока
файла нет, на его месте печатается заметная заглушка.

Запуск:
    python report/make_docx.py
    soffice --headless --convert-to pdf --outdir report/docx report/docx/*.docx
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
IMG = ROOT / "report" / "img"
SCREENS = IMG / "screens"
METRICS = ROOT / "report" / "metrics"
OUT = ROOT / "report" / "docx" / "Отчёт_ML_предмодерация_объявлений.docx"

REPO_URL = "https://github.com/Mapc27/devops-prime-era-1task"

# A4 минус поля: все таблицы и изображения нормируются под эту ширину,
# иначе на A4 часть таблиц вылезает за полосу набора
PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_L, MARGIN_R = 1.0, 0.8
USABLE = PAGE_W - MARGIN_L - MARGIN_R

BODY_FONT = "Times New Roman"
HEAD_FONT = "Arial"
# Courier New есть и в macOS, и в Windows — Consolas только в Windows
MONO_FONT = "Courier New"
ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
MUTED = RGBColor(0x55, 0x55, 0x55)
WARN = RGBColor(0xB0, 0x30, 0x30)

# Порядок рисунков в документе. Номер рисунка вычисляется из этого списка,
# поэтому ссылки в тексте не сбиваются при добавлении скриншотов.
FIGURES = [
    "architecture",
    "m1_thresholds",
    "scr_m1_cli",
    "scr_m1_api",
    "scr_m2_cli",
    "scr_m3_cli",
    "m4_stride",
    "m4_detection",
    "scr_m4_cli",
    "scr_ollama",
    "scr_m5_cli",
]

# что должно быть на каждом скриншоте — печатается в заглушке
SCREENSHOTS = {
    "scr_m1_cli": ("m1_cli.png",
                   "терминал: .venv/bin/python m1_text/predict.py --text "
                   "\"Не звоните мне, идиоты, в объявлении всё написано\" — "
                   "JSON с decision=reject, top_violation_ru=оскорбление, explained_by"),
    "scr_m1_api": ("m1_swagger.png",
                   "браузер: http://localhost:8000/docs после uvicorn m1_text.api:app — "
                   "раскрыт POST /predict, нажато Try it out → Execute, виден ответ 200"),
    "scr_m2_cli": ("m2_cli.png",
                   "терминал: m2_image/predict.py --image m4_video/data/frames/"
                   "garbage_truck_1.jpg --category \"Техника: ноутбуки\" — decision=reject и reason"),
    "scr_m3_cli": ("m3_cli.png",
                   "терминал: m3_audio/transcribe.py --audio m3_audio/data/audio/"
                   "09_clean.wav --moderate — язык, RTF, текст и вердикт M1"),
    "scr_m4_cli": ("m4_cli.png",
                   "терминал: m4_video/detect_video.py --video m4_video/data/video/"
                   "English_springer_1.mp4 --category \"Животные: собаки\" — "
                   "objects, declared_present=true, decision=allow"),
    "scr_ollama": ("ollama_list.png",
                   "терминал: ollama list и ollama ps — модели qwen2.5 установлены "
                   "и работают локально"),
    "scr_m5_cli": ("m5_cli.png",
                   "терминал: m5_llm/verdict.py --index 9 — входные результаты "
                   "проверок и JSON-ответ модели"),
}


# --------------------------------------------------------------------------- #
# метрики
# --------------------------------------------------------------------------- #
def load(name: str):
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def n(x: float, d: int = 2) -> str:
    """Число с десятичной запятой."""
    return f"{x:.{d}f}".replace(".", ",")


def pc(x: float, d: int = 1) -> str:
    """Доля -> проценты с десятичной запятой."""
    return f"{x * 100:.{d}f}".replace(".", ",") + " %"


def status(ok: bool) -> str:
    return "выполнен" if ok else "**не выполнен**"


M1 = load("m1_russian_toxicity_classifier_domain_final.json")
M1_TINY_DOMAIN = load("m1_rubert-tiny-toxicity_domain_fitted.json")
M1_TINY_PUBLIC = load("m1_rubert-tiny-toxicity_realrate.json")
M1_HEAVY_PUBLIC = load("m1_russian_toxicity_classifier_realrate.json")
M1_LAT_TINY = load("m1_single_rubert-tiny-toxicity.json")
M1_LAT_HEAVY = load("m1_single_russian_toxicity_classifier.json")
M1_LAT_CASCADE = load("m1_single_cascade.json")
M1_API = load("m1_api.json")
M2 = load("m2_image.json")
M3 = {r["model"]: r for r in load("m3_audio.json")}
M3_SNR = load("m3_snr.json")
M4 = load("m4_video.json")
M5 = {r["model"]: r for r in load("m5_llm.json")}

# ---- M1 ---------------------------------------------------------------- #
TH = M1["thresholds"]
B1 = M1["buckets"]
CV = M1["cv"]
KINDS = M1["by_kind"]
TINY_KINDS = M1_TINY_DOMAIN["by_kind"]
SWEEP = M1["recall_sweep"]
SWEEP_SEL = next(s for s in SWEEP if s["target_recall"] == TH["target_recall"])
SWEEP_MAX = max(SWEEP, key=lambda s: s["target_recall"])


def kinds_sum(label: int, key: str, kinds=KINDS) -> int:
    return sum(k[key] for k in kinds.values() if k["label"] == label)


N_CLEAN = kinds_sum(0, "n")
N_VIOL = kinds_sum(1, "n")
CLEAN = {b: kinds_sum(0, b) for b in ("allow", "review", "reject")}
VIOL = {b: kinds_sum(1, b) for b in ("allow", "review", "reject")}
CV_CLEAN = {b: kinds_sum(0, b, CV["by_kind"]) for b in ("allow", "review", "reject")}
CV_VIOL = {b: kinds_sum(1, b, CV["by_kind"]) for b in ("allow", "review", "reject")}
LAT_C = M1_LAT_CASCADE["latency_ms"]
API_LAT = M1_API["latency_ms"]
ROC_GAP = round((M1["roc_auc"] - M1_TINY_DOMAIN["roc_auc"]) * 100)
SPEED_RATIO = M1_LAT_HEAVY["latency_ms"]["p50"] / M1_LAT_TINY["latency_ms"]["p50"]

# ---- M2 ---------------------------------------------------------------- #
CLIP_NODIS = M2["clip_without_distractors"]
CLIP_DIS = M2["clip_with_distractors"]
RESNET = M2["resnet50_supervised"]
MM = M2["mismatch_detection"]
PERF = M2["perf_sweep"]
BATCHES = sorted(int(b) for b in PERF["mps"])
B_MIN, B_MAX = str(BATCHES[0]), str(BATCHES[-1])
GPU_GAIN = [PERF["cpu"][str(b)]["ms_per_image"] / PERF["mps"][str(b)]["ms_per_image"]
            for b in BATCHES]
BATCH_GAIN_MPS = PERF["mps"][B_MIN]["ms_per_image"] / PERF["mps"][B_MAX]["ms_per_image"]
BATCH_GAIN_CPU = PERF["cpu"][B_MIN]["ms_per_image"] / PERF["cpu"][B_MAX]["ms_per_image"]
UNIVERSALITY_PP = (RESNET["top1_accuracy"] - CLIP_DIS["top1_accuracy"]) * 100
DISTRACTOR_PP = (CLIP_NODIS["top1_accuracy"] - CLIP_DIS["top1_accuracy"]) * 100

# ---- M3 ---------------------------------------------------------------- #
W_SMALL = M3["whisper-small"]
W_SIZES = ["whisper-tiny", "whisper-base", "whisper-small"]
COND = {"clean": "чистая запись", "noisy15": "умеренный шум", "noisy10": "сильный шум"}
SNR = {c: M3_SNR[c]["snr_db_mean"] for c in ("noisy15", "noisy10")}
C_SMALL = W_SMALL["conditions"]
LOST_KEYWORD = next(e for e in W_SMALL["verdict_mismatches"] if e["audio"].endswith("noisy10.wav")
                    and e["expected"] == "reject")
CLEAN_FLIP = next((e for e in W_SMALL["verdict_mismatches"] if "_clean" in e["audio"]), None)

# ---- M4 ---------------------------------------------------------------- #
STRIDE = 5
Y_N = M4["models"]["yolo11n.pt"]
Y_S = M4["models"]["yolo11s.pt"]


def by_stride(model: dict, device: str) -> dict[int, dict]:
    return {r["stride"]: r for r in model["devices"][device]}


YN_MPS, YN_CPU = by_stride(Y_N, "mps"), by_stride(Y_N, "cpu")
YS_MPS, YS_CPU = by_stride(Y_S, "mps"), by_stride(Y_S, "cpu")
STRIDES = sorted(YN_MPS)
W4 = YN_MPS[STRIDE]
FULL4 = YN_MPS[1]
MAX_STRIDE = STRIDES[-1]
GPU_GAIN_4 = [YN_CPU[s]["elapsed_s"] / YN_MPS[s]["elapsed_s"] for s in STRIDES] + \
             [YS_CPU[s]["elapsed_s"] / YS_MPS[s]["elapsed_s"] for s in STRIDES]
CLIP_SECONDS = 6  # длительность тестовых роликов, m4_video/make_testset.py
SEC_30_MPS = W4["s_per_video"] * 30 / CLIP_SECONDS
SEC_30_CPU = YN_CPU[STRIDE]["s_per_video"] * 30 / CLIP_SECONDS

# ---- M5 ---------------------------------------------------------------- #
Q7, Q3 = M5["qwen2.5:7b-instruct"], M5["qwen2.5:3b-instruct"]
N5 = Q3["n_cases"]
N_CASES = N5 // Q3["repeats"]
Q7_ERR = len(Q7["mismatches"])
Q7_ERR_CASES = sorted({m["case"] for m in Q7["mismatches"]})
Q3_DEFECTS = len(Q3["text_defects"])
Q7_DEFECTS = len(Q7["text_defects"])
Q3_DEFECT_EXAMPLE = Q3["text_defects"][0]["text"] if Q3["text_defects"] else ""


# --------------------------------------------------------------------------- #
# инфраструктура оформления
# --------------------------------------------------------------------------- #
def set_font(run, name: str, size: float, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    # кириллица берёт шрифт из тех же полей, но их надо проставить явно
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = rpr.makeelement(qn("w:rFonts"), {})
        rpr.append(rfonts)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rfonts.set(qn(attr), name)


def add_marked(p, text: str, font: str, size: float, bold=False, italic=False,
               color=None):
    """Текст с разметкой **жирного** внутри строки.

    Звёздочки разбираются в любом месте строки, а не только по краям,
    иначе они попадают в документ как есть.
    """
    for i, chunk in enumerate(text.split("**")):
        if chunk:
            set_font(p.add_run(chunk), font, size, bold or i % 2 == 1, italic, color)


def para(doc, text="", size=12, bold=False, italic=False, align=None,
         space_after=6, space_before=0, font=BODY_FONT, color=None, indent=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.line_spacing = 1.15
    if align is not None:
        p.alignment = align
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    if text:
        add_marked(p, text, font, size, bold, italic, color)
    return p


def rich(doc, parts, size=12, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, indent=None):
    """Абзац из кусков [(текст, жирный, курсив), ...]."""
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    p.alignment = align
    if indent is not None:
        p.paragraph_format.left_indent = Inches(indent)
    for text, bold, italic in parts:
        add_marked(p, text, BODY_FONT, size, bold, italic)
    return p


def text(doc, s: str):
    """Обычный абзац; **жирное** размечается внутри строки."""
    return rich(doc, [(s, False, False)])


def h1(doc, s):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(s), HEAD_FONT, 16, bold=True, color=ACCENT)
    return p


def h2(doc, s):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(s), HEAD_FONT, 13, bold=True, color=ACCENT)
    return p


def h3(doc, s):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    set_font(p.add_run(s), HEAD_FONT, 11.5, bold=True)
    return p


def bullets(doc, items, size=12):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if isinstance(item, str):
            item = [(item, False, False)]
        for s, bold, italic in item:
            add_marked(p, s, BODY_FONT, size, bold, italic)


def shade(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.makeelement(qn("w:shd"), {})
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def repeat_header(row):
    """Шапка таблицы повторяется на каждой странице при переносе."""
    tr_pr = row._tr.get_or_add_trPr()
    el = tr_pr.makeelement(qn("w:tblHeader"), {})
    el.set(qn("w:val"), "true")
    tr_pr.append(el)


def table(doc, headers, rows, widths=None, size=10, header_fill="DCE6F1",
          align_right=()):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    repeat_header(t.rows[0])

    for i, s in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(2)
        set_font(p.add_run(str(s)), HEAD_FONT, size, bold=True)
        shade(cell, header_fill)

    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.space_before = Pt(2)
            if i in align_right:
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            add_marked(p, str(value), BODY_FONT, size)

    if widths:
        # ширины задаются в удобных пропорциях и масштабируются под полосу набора
        scale = (USABLE - 0.07) / sum(widths)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w * scale)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def fig(fid: str) -> int:
    return FIGURES.index(fid) + 1


def caption(doc, fid: str, s: str):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(12)
    set_font(cap.add_run(f"Рисунок {fig(fid)} — {s}"), BODY_FONT, 10,
             italic=True, color=MUTED)


def figure(doc, fid: str, path: Path, width_in: float, s: str):
    if not path.exists():
        para(doc, f"[изображение не найдено: {path.relative_to(ROOT)}]",
             italic=True, color=WARN)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        p.add_run().add_picture(str(path), width=Inches(min(width_in, USABLE - 0.1)))
    caption(doc, fid, s)


def screenshot(doc, fid: str, s: str, width_in: float = 6.0):
    """Скриншот запуска; пока файла нет — заметная заглушка с описанием."""
    filename, what = SCREENSHOTS[fid]
    path = SCREENS / filename
    if path.exists():
        figure(doc, fid, path, width_in, s)
        return
    p = para(doc, f"[СКРИНШОТ НЕ ДОБАВЛЕН — {what}. Положите файл "
                  f"report/img/screens/{filename} и пересоберите отчёт]",
             size=10, italic=True, color=WARN, align=WD_ALIGN_PARAGRAPH.CENTER,
             space_before=6, space_after=2)
    p.paragraph_format.keep_with_next = True
    caption(doc, fid, s)


def code(doc, lines):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.left_indent = Inches(0.3)
        p.paragraph_format.line_spacing = 1.0
        set_font(p.add_run(line), MONO_FONT, 9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


# --------------------------------------------------------------------------- #
# документ
# --------------------------------------------------------------------------- #
doc = Document()
sec = doc.sections[0]
sec.page_width = Inches(PAGE_W)
sec.page_height = Inches(PAGE_H)
sec.left_margin = Inches(MARGIN_L)
sec.right_margin = Inches(MARGIN_R)
sec.top_margin = Inches(0.8)
sec.bottom_margin = Inches(0.8)

style = doc.styles["Normal"]
style.font.name = BODY_FONT
style.font.size = Pt(12)
style.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)

footer_p = sec.footer.paragraphs[0]
footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
_fld = footer_p._p.makeelement(qn("w:fldSimple"), {qn("w:instr"): "PAGE"})
_run = footer_p._p.makeelement(qn("w:r"), {})
_txt = footer_p._p.makeelement(qn("w:t"), {})
_txt.text = "1"
_run.append(_txt)
_fld.append(_run)
footer_p._p.append(_fld)

# ------------------------------- титул ------------------------------------- #
para(doc, "Отчёт по лабораторной работе", size=19, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, font=HEAD_FONT, space_after=4)
para(doc, "Разработка приложений на основе готовых библиотек машинного обучения",
     size=13, align=WD_ALIGN_PARAGRAPH.CENTER, font=HEAD_FONT, space_after=20,
     color=ACCENT)
para(doc, "Тема: автоматическая предмодерация объявлений", size=14, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)

# --------------------------- 1. цель и условия ------------------------------ #
h1(doc, "1. Цель и условия работы")

text(doc, "Цель работы — разобраться, какие прикладные задачи сегодня "
          "решаются готовыми моделями машинного обучения, и построить "
          "работающее решение на пяти типах контента: тексте, аудио, "
          "изображениях, видео и локальной языковой модели.")

text(doc, "Работа велась в следующих условиях. Модели используются только "
          "предобученные: обучение и дообучение не проводились, в коде нет "
          "ни одной функции потерь, оптимизатора или обучающего цикла. Всё "
          "выполняется локально, без обращения к облачным провайдерам. "
          "Задействовано четыре разных фреймворка. Один из модулей заранее "
          "выбран кандидатом на последующую разработку API — им стал модуль "
          "обработки текста, поскольку он не требует файлов на входе "
          "и отвечает за миллисекунды.")

text(doc, "Весь стек открытый: библиотеки распространяются под свободными "
          "лицензиями, веса моделей взяты из открытых репозиториев, языковая "
          "модель развёрнута через Ollama на собственном оборудовании. "
          "Платные подписки и облачные API не использовались.")

h2(doc, "1.1. Выбор прикладной области")

text(doc, "Задание допускает как четыре несвязанные задачи, так и общую "
          "прикладную область. Я выбрал второе — предмодерацию объявлений "
          "на площадке частных объявлений, — потому что это даёт осмысленную "
          "систему метрик. В изолированной задаче можно сказать только "
          "«точность 0,9»; в прикладной области то же число превращается "
          "в «столько-то процентов потока уходит модератору, столько-то "
          "нарушений просачивается в публикацию». Второе проверяемо "
          "и обсуждаемо, первое — нет.")

text(doc, "На связность исполнения это не влияет: модули остаются "
          "независимыми и запускаются по отдельности. Общими у них являются "
          "только справочник категорий и модуль замеров.")

h2(doc, "1.2. Оборудование и окружение")

table(doc, ["Компонент", "Значение"], [
    ["Оборудование", "Apple M4 Pro, 24 ГБ оперативной памяти"],
    ["Операционная система", "macOS 26.6"],
    ["Язык и окружение", "Python 3.12, изолированное виртуальное окружение"],
    ["Ускоритель", "встроенный GPU через Metal (MPS); часть модулей сознательно "
                   "работает на процессоре — обоснование в разделах 5.1 и 5.3"],
], widths=[1.8, 4.8], size=10)

rich(doc, [("Все числа в отчёте получены прогонами скриптов из репозитория "
            "на этой конфигурации. Исходные машинные результаты сохранены "
            "в каталоге ", False, False), ("report/metrics/", False, True),
           (", и скрипт сборки отчёта читает измеренные значения прямо из этих "
            "файлов: после повторного прогона оценки отчёт пересобирается "
            "одной командой, без ручного переписывания таблиц.", False, False)])

# ------------------------------ 2. задача ----------------------------------- #
h1(doc, "2. Прикладная задача")
rich(doc, [("Площадка частных объявлений. Пользователь публикует объявление "
            "из четырёх типов контента: текстовое описание, фотографии, "
            "видеообзор и голосовой комментарий. Объём потока не позволяет "
            "проверять каждое объявление вручную. Каждый модуль относит "
            "проверяемый контент к одной из трёх корзин: ", False, False),
           ("allow", True, False), (" — публиковать, ", False, False),
           ("review", True, False), (" — модератору, ", False, False),
           ("reject", True, False), (" — отклонить.", False, False)])

h2(doc, "2.1. Цена ошибки определяет всю систему метрик")
rich(doc, [("Ошибки несимметричны. Ложная блокировка честного объявления "
            "обходится дороже пропущенного нарушения: продавец теряет деньги "
            "и уходит к конкуренту, тогда как пропуск обнаруживается жалобами "
            "и постмодерацией. Отсюда жёсткое ограничение — ", False, False),
           ("точность на классе reject не ниже 0,95", True, False),
           (", — а полнота максимизируется уже под ним. Поэтому корзин три: "
            "всё, в чём модель не уверена, передаётся человеку, а не "
            "блокируется.", False, False)])
text(doc, "Это же правило определило архитектуру: итоговый вердикт вычисляет "
          "детерминированное правило в коде, а языковая модель только "
          "формулирует объяснение. Решение о блокировке — не то место, где "
          "уместна вероятностная генерация.")

# --------------------------- 3. архитектура --------------------------------- #
h1(doc, "3. Архитектура и обоснование стека")
figure(doc, "architecture", IMG / "architecture.png", 6.3,
       "Пять модулей, используемые библиотеки, готовые модели "
       "и измеренные показатели")

h2(doc, "3.1. Архитектурный стиль")
text(doc, "Выбран стиль независимых модулей с общими справочными данными, "
          "а не единого конвейера: по условию это пять самостоятельных задач, "
          "отказ одного модуля не должен останавливать остальные, а стоимость "
          "запуска различается от миллисекунды на текст до секунд на видео — "
          "связывать такое в синхронный конвейер значило бы равнять всё "
          "по самому медленному.")
rich(doc, [("Общим сделан только справочник категорий ", False, False),
           ("catalog.json", False, True),
           (" — это данные, а не исполнение. До его введения у модулей были "
            "собственные списки категорий, противоречившие друг другу.",
            False, False)])

h2(doc, "3.2. Обоснование выбора библиотек")
table(doc, ["Модуль", "Библиотека", "Почему именно она"], [
    ["M1, M2", "Hugging Face\nTransformers",
     "Препроцессинг поставляется вместе с весами: токенизатор и нормализация "
     "те же, что при обучении. Ручная реализация расходится в мелочах, и потеря "
     "качества остаётся незаметной. Библиотека также сообщает устройство модели "
     "через config, что позволило написать модуль, работающий и с multilabel-, "
     "и с бинарными классификаторами без правок"],
    ["M3", "faster-whisper\n(CTranslate2)",
     "Квантизация int8 из коробки: модель small работает на процессоре "
     "в несколько раз быстрее реального времени. Встроенная фильтрация тишины "
     "ускоряет обработку и снижает число галлюцинаций"],
    ["M4", "Ultralytics",
     "Берёт на себя чтение видео, масштабирование, подавление дублирующих рамок "
     "и отрисовку. Пропуск кадров выполняется на уровне загрузчика — "
     "пропущенные кадры не декодируются вовсе"],
    ["M5", "Ollama\n(llama.cpp)",
     "Установка одной командой, использование Metal и, главное, "
     "структурированный вывод: передача JSON-схемы ограничивает генерацию так, "
     "что невалидный ответ невозможен"],
], widths=[0.7, 1.4, 4.5], size=10)

# ------------------------------ 4. ход работы ------------------------------- #
h1(doc, "4. Ход работы и зоны ответственности")
table(doc, ["Этап", "Содержание"], [
    ["1", "Постановка задачи и ТЗ: выбрана прикладная область, определена цена "
          "ошибки, из неё выведены критерии приёмки и метрики"],
    ["2", "Окружение: Python 3.12 и ffmpeg через Homebrew, изолированное "
          "виртуальное окружение, глобальный интерпретатор не затронут"],
    ["3", "Модуль текста: инференс, подбор порогов, оценка, HTTP-обёртка. "
          "Выбран кандидатом на API — не требует файлов на входе"],
    ["4-7", "Модули изображений, аудио, видео и языковой модели: по каждому "
            "скрипт инференса и скрипт оценки с сохранением метрик"],
    ["8", "Унификация справочника категорий, сравнение моделей разного размера, "
          "замеры на процессоре и ускорителе"],
    ["9", "Расширение тестового набора для текста с 60 до 200 примеров "
          "с разбивкой по видам нарушений — этот шаг изменил вывод по модулю"],
    ["10", "Проверка результатов: кросс-валидация порогов текста, пересчёт "
           "метрики подмены категории для фото по рабочей логике решений, "
           "автоматическая проверка текста объяснений LLM"],
    ["11", "Отчётность: все замеры сохраняются машиночитаемо, отчёт собирается "
           "из них скриптом"],
], widths=[0.5, 6.1], size=10)
rich(doc, [("Единый модуль ", False, False), ("common/bench.py", False, True),
           (" измеряет латентность, память и устройство одинаково во всех "
            "модулях — иначе цифры разных модулей несопоставимы. Результаты "
            "сохраняются в ", False, False), ("report/metrics/*.json", False, True),
           (", отчёт собирается из них, а не переписывается руками.",
            False, False)])
text(doc, "Работа выполнена индивидуально; распределение ролей между "
          "участниками не производилось.")

# ------------------------------ 5. модули ----------------------------------- #
doc.add_page_break()
h1(doc, "5. Модули: реализация и результаты")
text(doc, "По каждому модулю приводятся пять обязательных разделов: "
          "преимущества библиотеки, принцип работы модели и причины выбора, "
          "структура датасета, метрики качества, особенности реализации "
          "и эффективность. Сравнения моделей разного размера вынесены "
          "в раздел 6, сводка по критериям приёмки — в раздел 7.")

# ---------------------------------- M1 -------------------------------------- #
h2(doc, "5.1. M1 — текст: детекция нарушений в описании")

h3(doc, "Преимущества библиотеки")
text(doc, "Transformers сводит задачу к трём операциям: загрузка весов "
          "по идентификатору, токенизация тем же токенизатором, которым "
          "модель обучалась, получение логитов. Дополнительно библиотека "
          "сообщает устройство модели — состав классов и тип задачи. Модуль "
          "читает эти поля и работает с моделями разного устройства без "
          "правок кода; без этого сравнить две модели было бы невозможно.")

h3(doc, "Принцип работы модели и причины выбора")
text(doc, "Используются две модели в каскаде, и к этому привёл замер.")
bullets(doc, [
    [("Решение принимает ", False, False),
     ("s-nlp/russian_toxicity_classifier", False, True),
     (" — энкодер на базе rubert-base, 178 млн параметров, бинарная голова. "
      "Текст разбивается на токены, слои self-attention набирают контекст, "
      "итоговое представление идёт в линейный слой.", False, False)],
    [("Вид нарушения добавляет ", False, False),
     ("cointegrated/rubert-tiny-toxicity", False, True),
     (" — 11,8 млн параметров, 45 МБ, пять независимых сигмоид: отсутствие "
      "нарушений, оскорбление, нецензурная лексика, угроза, опасный контент.",
      False, False)],
])
text(doc, f"Изначально взята была только лёгкая модель. На публичном корпусе "
          f"она давала ROC-AUC {n(M1_TINY_PUBLIC['roc_auc'], 3)} и выглядела "
          f"достаточной. На собственном наборе объявлений выяснилось, что она "
          f"детектирует только прямые оскорбления с бранными существительными, "
          f"а угрозы и презрение пропускает почти полностью. Тяжёлая модель эти "
          f"виды ловит, но бинарная и не сообщает, что именно нарушено, — а это "
          f"требуется для формулировки причины отказа. Отсюда каскад: решение "
          f"по тяжёлой модели, вид нарушения по лёгкой и только для уже "
          f"помеченных текстов. На чистом потоке, а это около 95 % объявлений, "
          f"вторая модель не запускается вовсе.")

h3(doc, "Структура датасета")
text(doc, f"Обучение: модели взяты готовыми, обучались на открытых корпусах "
          f"русских токсичных комментариев. Для оценки используются два "
          f"набора: публичный корпус (6000 текстов, 17,2 % нарушений) даёт "
          f"устойчивые метрики, но модели обучались на данных того же "
          f"происхождения, поэтому результат на нём — оценка сверху. "
          f"Собственный набор объявлений ({M1['n_samples']} текстов) модели "
          f"не видели.")
text(doc, "Собственный набор строился не как «чистые против грубых», "
          "а по подгруппам, и именно это сделало его информативным:")
KIND_DESC = {
    "plain": "обычные нейтральные объявления",
    "firm": "законные резкие формулировки: «перекупщиков не беспокоить»",
    "negative": "резкие оценки товара, но не людей: «состояние ужасное»",
    "insult": "прямые оскорбления покупателей",
    "contempt": "презрение и дискриминация, без бранных слов",
    "threat": "угрозы, сформулированные без брани",
    "borderline": "грубость без оскорбительных существительных",
}
table(doc, ["Подгруппа", "n", "Содержание"],
      [[k, str(KINDS[k]["n"]), d] for k, d in KIND_DESC.items()],
      widths=[1.1, 0.4, 5.1], size=10, align_right=(1,))
text(doc, "Подгруппы firm и negative — проверка на ложные срабатывания: такие "
          "объявления публикуются ежедневно и блокировать их нельзя. "
          "Подгруппы contempt, threat и borderline — проверка на покрытие.")

h3(doc, "Метрики качества: покрытие видов нарушений")
text(doc, "Первый вопрос к модели — видит ли она вообще тот или иной вид "
          "нарушения. Ответ на него не должен зависеть от выбранного порога, "
          "поэтому сравнение ведётся по медианной оценке внутри подгруппы: "
          "это характерное число, которое модель выдаёт на таких текстах.")
KIND_TITLE = {
    "insult": "insult — прямые оскорбления",
    "borderline": "borderline — грубость без брани",
    "threat": "threat — угрозы",
    "contempt": "contempt — презрение",
    "plain": "plain — обычные объявления",
    "firm": "firm — резкие, но законные",
    "negative": "negative — негатив о товаре",
}
table(doc, ["Подгруппа", "n", "rubert-tiny (11,8M)", "russian_toxicity (178M)"],
      [[t, str(KINDS[k]["n"]), n(TINY_KINDS[k]["median_score"], 4),
        (f"**{n(KINDS[k]['median_score'], 4)}**" if KINDS[k]["label"]
         else n(KINDS[k]["median_score"], 4))]
       for k, t in KIND_TITLE.items()],
      widths=[2.4, 0.4, 1.9, 1.9], size=10, align_right=(1, 2, 3))
hidden = [TINY_KINDS[k]["median_score"] for k in ("threat", "contempt", "borderline")]
text(doc, f"Лёгкая модель отделяет от чистых текстов только прямые "
          f"оскорбления: её медиана на них {n(TINY_KINDS['insult']['median_score'], 4)} "
          f"против {n(TINY_KINDS['plain']['median_score'], 4)} на обычных "
          f"объявлениях — разрыв в три порядка. На угрозах, презрении "
          f"и грубости без брани медианы {n(min(hidden), 4)}–{n(max(hidden), 4)}, "
          f"то есть ровно того же порядка, что у чистых текстов. **Никакой порог "
          f"не отделит одно от другого**, потому что модель их не различает.")
text(doc, "Это видно и напрямую. На угрозах её специализированная голова "
          "threat выдаёт ноль:")
code(doc, [
    "«Если не придёшь на встречу — найду и проблемы будут, понял?»",
    "   non-toxic=1.000  insult=0.000  obscenity=0.000  threat=0.000",
])
text(doc, "Причина в обучающих данных: форумные корпуса токсичности содержат "
          "прямую брань, а деловая угроза в объявлении опасна смыслом, "
          "а не словами. Наличие метки threat в конфигурации модели ничего "
          "не гарантирует.")
clean_meds = [KINDS[k]["median_score"] for k in ("plain", "firm", "negative")]
text(doc, f"У тяжёлой модели разрыв между нарушениями и чистыми текстами "
          f"сохраняется на всех подгруппах: {n(KINDS['borderline']['median_score'], 4)} "
          f"и {n(KINDS['threat']['median_score'], 4)} против "
          f"{n(min(clean_meds), 4)}–{n(max(clean_meds), 4)}. Разрыв невелик, "
          f"но он есть, и порог его использует. Именно из-за того, что он "
          f"невелик, рабочие пороги получаются очень низкими — это главное "
          f"слабое место модуля, о нём ниже.")

table(doc, ["Модель", "Публичный корпус", "Объявления", "p50 на текст"], [
    ["rubert-tiny, 11,8M", f"ROC-AUC {n(M1_TINY_PUBLIC['roc_auc'], 4)}",
     n(M1_TINY_DOMAIN["roc_auc"], 4), f"**{n(M1_LAT_TINY['latency_ms']['p50'])} мс**"],
    ["russian_toxicity, 178M", f"ROC-AUC {n(M1_HEAVY_PUBLIC['roc_auc'], 4)}",
     f"**{n(M1['roc_auc'], 4)}**", f"{n(M1_LAT_HEAVY['latency_ms']['p50'])} мс"],
    ["каскад", "—", "—", f"{n(LAT_C['p50'], 1)} мс (p95 {n(LAT_C['p95'], 1)})"],
], widths=[2.2, 1.6, 1.3, 1.5], size=10, align_right=(1, 2, 3))
text(doc, f"ROC-AUC измеряет качество ранжирования и от выбора порога "
          f"не зависит. На публичном корпусе модели неразличимы, на домене "
          f"объявлений разрыв {ROC_GAP} пунктов.")

h3(doc, "Выбор порогов")
text(doc, f"Модель выдаёт не решение, а оценку от нуля до единицы. Решение "
          f"получается сравнением этой оценки с порогами, и порог — это наш "
          f"выбор, а не свойство модели. Корзин три, значит порогов два: "
          f"ниже нижнего объявление публикуется, выше верхнего отклоняется, "
          f"между ними уходит модератору. Ниже описано, как эти два числа "
          f"получены; все три панели рисунка {fig('m1_thresholds')} относятся "
          f"к этому выбору.")

rich(doc, [("Шаг 1. Что требуется. ", True, False),
           ("Из цены ошибки (раздел 2.1) следуют два ограничения: точность "
            "на классе reject не ниже 0,95 и нагрузка на модераторов не выше "
            "20 % потока. Полнота максимизируется при соблюдении обоих.",
            False, False)])

rich(doc, [("Шаг 2. Поправка на базовую частоту. ", True, False),
           (f"Точность нельзя брать прямо с размеченной выборки: в ней "
            f"{pc(M1['positive_rate'], 0)} нарушений, а в потоке ожидается около "
            f"{pc(TH['assumed_violation_rate'], 0)}. Точность от этой доли "
            "зависит, полнота и доля ложных срабатываний — нет, поскольку "
            "последние две считаются внутри своего класса. Поэтому точность "
            "пересчитывается аналитически:", False, False)])
code(doc, ["precision(p) = p · TPR / ( p · TPR + (1 − p) · FPR )"])
text(doc, "где p — ожидаемая доля нарушений, TPR — доля пойманных нарушений, "
          "FPR — доля ошибочно помеченных чистых объявлений. Физически "
          "пересэмплировать набор было бы расточительно: чтобы из 100 на 100 "
          "получить 5 %, пришлось бы выбросить 95 нарушений из ста.")
text(doc, f"Эффект показан на средней панели рисунка {fig('m1_thresholds')}. "
          f"Синяя кривая — точность, посчитанная прямо по выборке; зелёная — "
          f"она же в пересчёте на поток с 5 % нарушений. При одной и той же "
          f"полноте зелёная проходит ниже: в реальном потоке те же ложные "
          f"срабатывания весят больше, потому что чистых объявлений "
          f"значительно больше. Выбирать порог по синей кривой значило бы "
          f"получить в эксплуатации точность ниже заявленной.")

rich(doc, [("Шаг 3. Верхний порог. ", True, False),
           (f"Берётся минимальное значение, при котором зелёная кривая "
            f"достигает требуемых 0,95; среди подходящих выбирается то, что "
            f"даёт наибольшую полноту. Получается {n(TH['reject'], 4)} — "
            f"красная точка на средней панели: полнота {n(B1['recall_reject'])} "
            f"при точности {n(B1['precision_reject'])}. На левой панели эта же "
            f"точка лежит в левом нижнем углу ROC-кривой: доля ложных "
            f"срабатываний на этой выборке равна нулю.", False, False)])

rich(doc, [("Шаг 4. Нижний порог. ", True, False),
           (f"Он отвечает на другой вопрос: какую долю нарушений система "
            f"обязана хотя бы показать человеку. Требование «не менее "
            f"{pc(SWEEP_MAX['target_recall'], 0)}» даёт порог "
            f"{n(SWEEP_MAX['thresholds']['review'], 4)}, но тогда модератору уходит "
            f"{pc(SWEEP_MAX['buckets']['share_review'])} потока — больше "
            f"допустимых 20 %. Перебор целевого значения показывает цену "
            f"размена:", False, False)])


def sweep_row(s):
    b = s["buckets"]
    load_s = pc(b["share_review"]) + (" — сверх лимита" if b["share_review"] > 0.2 else "")
    row = [n(s["target_recall"]), n(s["thresholds"]["review"], 4), load_s,
           pc(b["violations_leaked_to_allow"]), pc(b["clean_wrongly_rejected"], 0)]
    return [f"**{c}**" for c in row] if s is SWEEP_SEL else row


table(doc, ["Цель по полноте", "Порог review", "Нагрузка на модераторов",
            "Утечка в allow", "Ложные отклонения"],
      [sweep_row(s) for s in SWEEP],
      widths=[1.2, 1.0, 2.0, 1.2, 1.2], size=10, align_right=(0, 1, 3, 4))
text(doc, f"Выбрана строка {n(SWEEP_SEL['target_recall'])}: нагрузка "
          f"{pc(SWEEP_SEL['buckets']['share_review'])} укладывается в ограничение. "
          f"Это осознанный размен по правилу цены ошибки — жертвуем полнотой, "
          f"а не точностью, потому что пропуск нарушения дешевле ложной "
          f"блокировки. Итоговая пара порогов: {n(TH['review'], 4)} "
          f"и {n(TH['reject'], 4)}.")

figure(doc, "m1_thresholds", IMG / "m1_russian_toxicity_classifier_domain_final.png", 6.3,
       "Выбор порогов. Слева: ROC-кривая с отмеченными рабочими "
       "точками обоих порогов. В центре: точность против полноты на выборке "
       "и в пересчёте на реальную долю нарушений. Справа: распределение "
       "объявлений по трём корзинам при выбранных порогах")

h3(doc, "Рабочая точка")
text(doc, f"Правая панель рисунка {fig('m1_thresholds')} показывает итог — как "
          f"делится поток. Обычная матрица ошибок здесь не подходит: она бинарна "
          f"и схлопывает корзину ручной проверки, которой в системе отведена "
          f"отдельная роль.")
table(doc, ["", "allow — публикуем", "review — модератору", "reject — отклоняем"], [
    [f"Чистые объявления ({N_CLEAN})", f"**{CLEAN['allow']}**", str(CLEAN["review"]),
     f"**{CLEAN['reject']}**"],
    [f"Нарушения ({N_VIOL})", str(VIOL["allow"]), str(VIOL["review"]), str(VIOL["reject"])],
], widths=[2.2, 1.5, 1.5, 1.5], size=10, align_right=(1, 2, 3))
text(doc, f"На этой выборке ни одно из {N_CLEAN} честных объявлений не отклонено. "
          f"{VIOL['allow']} нарушений из {N_VIOL} прошли в публикацию: это и есть "
          f"плата за выбор точки {n(SWEEP_SEL['target_recall'])}.")

text(doc, "В разрезе видов нарушений картина такая:")
table(doc, ["Подгруппа", "n", "reject", "review", "allow"],
      [[KIND_TITLE[k], str(KINDS[k]["n"]), str(KINDS[k]["reject"]),
        str(KINDS[k]["review"]), str(KINDS[k]["allow"])]
       for k in ("insult", "borderline", "threat", "contempt")] +
      [["чистые объявления", str(N_CLEAN), f"**{CLEAN['reject']}**",
        str(CLEAN["review"]), str(CLEAN["allow"])]],
      widths=[2.6, 0.5, 1.1, 1.1, 1.1], size=10, align_right=(1, 2, 3, 4))
text(doc, f"Прямые оскорбления отклоняются все. Угрозы и грубость "
          f"без брани в основном попадают к модератору, а не отклоняются "
          f"автоматически: оценки модели на них невысоки, и автоматический отказ "
          f"по таким оценкам означал бы риск для честных объявлений. Хуже всего "
          f"с презрением и дискриминацией: {KINDS['contempt']['allow']} случаев "
          f"из {KINDS['contempt']['n']} уходят в публикацию. Пропускаются и "
          f"отдельные угрозы ({KINDS['threat']['allow']} из "
          f"{KINDS['threat']['n']}) — в том числе приведённая выше фраза "
          f"«найду и проблемы будут»: её оценка тяжёлой моделью ниже порога review.")

h3(doc, "Проверка порогов на отложенных данных")
text(doc, f"Пороги выше подобраны на тех же {M1['n_samples']} текстах, на которых "
          f"посчитаны метрики, поэтому эти метрики оптимистичны: порог "
          f"подстраивается под конкретные примеры. Чтобы оценить, как пороги "
          f"поведут себя на новых объявлениях, проведена {CV['k']}-кратная "
          f"кросс-валидация: набор делится на {CV['k']} частей с сохранением "
          f"долей подгрупп, пороги подбираются на {CV['k'] - 1} частях тем же "
          f"алгоритмом, а решение по каждому тексту принимают пороги, которые "
          f"этот текст не видели.")
B_CV = CV["buckets"]
table(doc, ["Показатель", "Подбор и проверка на одних данных",
            f"Кросс-валидация, {CV['k']} частей"], [
    ["precision(reject) на выборке 50/50", n(B1["precision_reject"], 3),
     n(B_CV["precision_reject"], 3)],
    [f"precision(reject) в потоке с {pc(B_CV['prior'], 0)} нарушений", "1,000",
     f"**{n(B_CV['precision_reject_at_prior'], 3)}**"],
    ["Полнота reject", n(B1["recall_reject"]), n(B_CV["recall_reject"])],
    ["Нагрузка на модераторов", pc(B1["share_review"]), pc(B_CV["share_review"])],
    ["Утечка нарушений в allow", pc(B1["violations_leaked_to_allow"]),
     pc(B_CV["violations_leaked_to_allow"])],
    ["Честных объявлений отклонено", f"{CLEAN['reject']} из {N_CLEAN}",
     f"**{CV_CLEAN['reject']} из {N_CLEAN}**"],
    ["Разброс порога reject по частям", "—",
     f"{n(CV['threshold_range']['reject'][0], 4)}–{n(CV['threshold_range']['reject'][1], 4)}"],
], widths=[2.6, 2.0, 2.0], size=10, align_right=(1, 2))
text(doc, f"Кросс-валидация меняет вывод. На отложенных данных "
          f"{CV_CLEAN['reject']} честных объявления из {N_CLEAN} отклоняются, "
          f"и в пересчёте на поток с 5 % нарушений точность отказа падает "
          f"до {n(B_CV['precision_reject_at_prior'], 2)} — **требование 0,95 "
          f"на новых данных не подтверждается**. Причина видна в последней "
          f"строке: порог reject между частями меняется в несколько раз, то есть "
          f"ста чистых примеров недостаточно, чтобы надёжно поставить порог "
          f"в области, где модель выдаёт оценки порядка сотых. Нижний порог "
          f"({n(TH['review'], 4)}) ещё чувствительнее: сдвиг пунктуации меняет "
          f"оценку на тысячные, и вердикт переходит из review в allow — такой "
          f"случай найден в модуле аудио (раздел 5.3).")
text(doc, "Вывод для эксплуатации: автоматический отказ по этим порогам "
          "допустим только после калибровки на размеченных объявлениях "
          "реального потока объёмом в тысячи примеров. До этого безопаснее "
          "направлять в review всё, что выше нижнего порога.")

h3(doc, "Продуктовые и инфраструктурные метрики")
table(doc, ["Показатель", "Значение"], [
    ["Доля автоматических решений", pc(B1["automated_share"])],
    ["Передано модераторам", pc(B1["share_review"])],
    ["Честных объявлений ошибочно отклонено",
     f"{CLEAN['reject']} из {N_CLEAN} (кросс-валидация: {CV_CLEAN['reject']} из {N_CLEAN})"],
    ["Латентность каскада, p50 / p95", f"**{n(LAT_C['p50'], 1)} / {n(LAT_C['p95'], 1)} мс** (CPU)"],
    ["Латентность HTTP-сервиса, p50 / p95",
     f"{n(API_LAT['p50'], 1)} / {n(API_LAT['p95'], 1)} мс"],
    ["Пропускная способность, один поток", f"{M1_LAT_CASCADE['throughput_per_s']:.0f} текстов/с"],
    ["Пиковая оперативная память", f"{M1_LAT_CASCADE['memory_mb']['rss_peak']:.0f} МБ"],
], widths=[3.3, 3.3], size=10)
text(doc, f"Критерий по латентности (p95 менее 100 мс) выполнен с запасом: "
          f"p95 каскада {n(LAT_C['p95'], 1)} мс, сервиса вместе с разбором HTTP — "
          f"{n(API_LAT['p95'], 1)} мс.")

h3(doc, "Особенности реализации и эффективность")
bullets(doc, [
    [("Каскад с ленивой загрузкой в CLI: ", True, False),
     ("модель-объяснитель создаётся при первом нарушении, на чистом потоке "
      "не загружается вообще. В HTTP-сервисе, наоборот, обе модели грузятся "
      f"и прогреваются при старте: ленивая загрузка давала "
      f"{n(LAT_C['max'] / 1000, 1)} с на первом запросе с нарушением.", False, False)],
    [("Система не выдумывает причину отказа: ", True, False),
     ("вид нарушения называется только при уверенности выше 0,5, иначе "
      "объявление помечается как требующее ручной проверки. Без этого порога "
      "выбор максимума всегда что-нибудь возвращает — у лёгкой модели голова "
      "«опасный контент» срабатывает даже на текстах, которые она считает "
      "чистыми.", False, False)],
    [("Работа на процессоре при доступном ускорителе: ", True, False),
     ("для текста в 20-40 токенов перенос тензоров на GPU дороже вычислений.",
      False, False)],
    [("Пороги отделены от кода ", True, False),
     ("и несут пометку, на каком датасете и какой моделью подобраны, плюс "
      "предупреждение о непереносимости между доменами.", False, False)],
    [("HTTP-обёртка ", True, False),
     (f"на FastAPI: эндпоинты /health, /predict и /predict/batch, валидация "
      f"входа по длине текста. Старт с загрузкой и прогревом обеих моделей — "
      f"{n(M1_API['startup_s'], 1)} с, далее {n(API_LAT['p50'], 1)} мс на запрос "
      f"(p50) вместе с разбором HTTP.", False, False)],
])
screenshot(doc, "scr_m1_cli", "Запуск M1 из командной строки: текст с оскорблением "
                              "отклонён, вид нарушения определён второй ступенью каскада")
screenshot(doc, "scr_m1_api", "HTTP-сервис M1: интерактивная документация FastAPI "
                              "и ответ на запрос POST /predict", width_in=4.6)

# ---------------------------------- M2 -------------------------------------- #
h2(doc, "5.2. M2 — изображение: соответствие фото категории")

h3(doc, "Преимущества библиотеки")
text(doc, "Та же Transformers, но здесь важно другое свойство: процессор "
          "модели содержит ровно тот препроцессинг, с которым она обучалась — "
          "размер, кадрирование, нормализацию по каналам. Ошибка в любом "
          "из этих чисел даёт молчаливую деградацию: модель работает, "
          "точность просто ниже.")

h3(doc, "Принцип работы модели и причины выбора")
text(doc, "CLIP ViT-B/32 (151 млн параметров, 577 МБ) — две башни: Vision "
          "Transformer для изображений и трансформер для текста, обе проецируют "
          "вход в общее пространство на 512 измерений. Обучение шло на парах "
          "«картинка — подпись»: эмбеддинги верной пары сближались, неверной "
          "расходились. Следствие: чтобы классифицировать изображение, "
          "достаточно описать классы словами.")
text(doc, "Причина выбора следует из задачи: категорий на площадке сотни, "
          "они добавляются и переименовываются, обучать классификатор "
          "под каждую нельзя. Новая категория — несколько строк в справочнике, "
          "а не цикл переобучения.")

h3(doc, "Структура датасета")
text(doc, f"Обучение: около 400 млн пар «изображение — подпись» из интернета. "
          f"Инференс: файл JPEG или PNG плюс заявленная категория из справочника "
          f"на 23 позиции. Оценка: Imagenette, валидационный сплит, "
          f"{M2['n_samples']} изображений, 10 классов с готовой разметкой; классы "
          f"интерпретируются как товарные категории. Для измерения детекции "
          f"подмены каждое изображение проверяется дважды: с верной категорией "
          f"(честное объявление) и с заведомо неверной (подмена). В справочник "
          f"включены три «дистрактора» — скриншот, документ, селфи: типовые "
          f"непригодные вложения, которые модель должна распознавать, а не "
          f"натягивать на товарную категорию.")

h3(doc, "Метрики качества")
table(doc, ["Модель", "Вариантов выбора", "top-1", "top-3"], [
    ["CLIP zero-shot", "23 категории", n(CLIP_NODIS["top1_accuracy"], 3),
     n(CLIP_NODIS["top3_accuracy"], 3)],
    ["CLIP zero-shot", "23 + 3 дистрактора", f"**{n(CLIP_DIS['top1_accuracy'], 3)}**",
     n(CLIP_DIS["top3_accuracy"], 3)],
    ["ResNet50 обученный", "10 классов", f"**{n(RESNET['top1_accuracy'], 3)}**", "—"],
], widths=[1.8, 1.9, 1.4, 1.4], size=10, align_right=(2, 3))
text(doc, f"Цена универсальности измерена: {n(UNIVERSALITY_PP, 1)} процентного "
          f"пункта. Сравнение поставлено в пользу обученной модели — ResNet50 "
          f"знает ровно эти 10 классов и его выход из 1000 сужен до нужных, "
          f"а CLIP выбирает из 26 вариантов, 16 из которых в тестовом наборе "
          f"вообще не встречаются. Добавление трёх дистракторов стоило CLIP "
          f"{n(DISTRACTOR_PP, 1)} пункта: чем больше вариантов в справочнике, "
          f"тем больше поводов ошибиться.")
text(doc, "Основной сценарий модуля — решение по заявленной категории. "
          "Оно принимается той же функцией, что и в рабочем коде: allow, если "
          "распознана заявленная категория с уверенностью не ниже 0,5; "
          "review, если модель ни в чём не уверена; reject, если модель "
          "уверенно видит другую категорию.")
table(doc, ["", "allow", "review", "reject"], [
    ["Честные объявления", f"**{pc(MM['honest']['allow'])}**", pc(MM["honest"]["review"]),
     pc(MM["honest"]["reject"])],
    ["Подменённая категория", pc(MM["swapped"]["allow"]), pc(MM["swapped"]["review"]),
     f"**{pc(MM['swapped']['reject'])}**"],
], widths=[2.2, 1.5, 1.5, 1.5], size=10, align_right=(1, 2, 3))
text(doc, f"Подмена не проходит в автоматическую публикацию в "
          f"{pc(MM['recall_mismatch'])} случаев, из них в "
          f"{pc(MM['recall_mismatch_reject'])} она отклоняется автоматически. "
          f"Точность отказа — {n(MM['precision_reject'], 3)} при равной доле "
          f"честных и подменённых объявлений; честных фотографий ошибочно "
          f"отклонено {pc(MM['false_reject_rate'])}. Это часть случаев, где "
          f"модель неверно распознала класс (ошибка top-1 — "
          f"{pc(1 - CLIP_DIS['top1_accuracy'])}); остальные неуверенные "
          f"распознавания порог отправил в review, а не в отказ.")
text(doc, "Оговорка: неверная категория назначается сдвигом по справочнику, "
          "и чаще всего она визуально далека от истинной (собака против "
          "бензопилы). Подмена на близкую категорию — «Автомобили» вместо "
          "«Спецтехники» — может ловиться заметно хуже; такой набор в работе "
          "не собирался, и приведённые цифры относятся только к грубой подмене.")
table(doc, ["Размер батча", "GPU (MPS), мс/изобр.", "CPU, мс/изобр.", "Выигрыш GPU"],
      [[str(b), n(PERF["mps"][str(b)]["ms_per_image"]), n(PERF["cpu"][str(b)]["ms_per_image"]),
        f"{n(g)}×"] for b, g in zip(BATCHES, GPU_GAIN)],
      widths=[1.4, 2.0, 1.6, 1.4], size=10, align_right=(0, 1, 2, 3))
text(doc, f"Батчинг даёт больше, чем GPU: переход к батчу из {B_MAX} ускоряет "
          f"в {n(BATCH_GAIN_MPS, 1)} раза на GPU и в {n(BATCH_GAIN_CPU, 1)} раза "
          f"на процессоре, тогда как переход с процессора на GPU — "
          f"в {n(min(GPU_GAIN), 1)}–{n(max(GPU_GAIN), 1)} раза. Если поток "
          f"фотографий позволяет накапливать пачки, это выгоднее, чем требовать "
          f"ускоритель.")

h3(doc, "Особенности реализации и эффективность")
bullets(doc, [
    [("Эмбеддинги категорий считаются один раз при инициализации: ", True, False),
     ("тексты категорий не меняются от запроса к запросу, поэтому на запрос "
      "приходится только кодирование изображения и одно матричное умножение.",
      False, False)],
    [("Промпт-ансамблирование: ", True, False),
     ("на категорию три формулировки, их нормированные эмбеддинги усредняются — "
      "снижается зависимость от одной неудачной формулировки.", False, False)],
    [("Температура берётся из модели, ", True, False),
     ("а не подбирается: она обучена вместе с весами.", False, False)],
    [("Одна функция решения для инференса и оценки: ", True, False),
     ("метрика подмены считается тем же кодом, что принимает решение "
      "в эксплуатации, поэтому не может разойтись с ним.", False, False)],
])
screenshot(doc, "scr_m2_cli", "Запуск M2: на фото мусоровоз, заявлена категория "
                              "«Техника: ноутбуки» — объявление отклонено с причиной")

# ---------------------------------- M3 -------------------------------------- #
h2(doc, "5.3. M3 — аудио: распознавание речи")

h3(doc, "Преимущества библиотеки")
text(doc, f"faster-whisper — реализация Whisper на движке CTranslate2 вместо "
          f"PyTorch с квантизацией int8 из коробки. Существенное ограничение "
          f"платформы: CTranslate2 не поддерживает Metal, модель работает "
          f"на процессоре. На использованном оборудовании это не проблема — "
          f"в int8 отношение времени обработки к длительности записи "
          f"(RTF) составляет {n(C_SMALL['clean']['rtf_mean'])}. Ограничение "
          f"зафиксировано явно, потому что при переносе на машину с CUDA "
          f"картина изменится.")

h3(doc, "Принцип работы модели и причины выбора")
text(doc, "Whisper small (244 млн параметров, 461 МБ в int8) — трансформер "
          "кодировщик-декодировщик. Аудио приводится к 16 кГц моно "
          "и превращается в лог-мел-спектрограмму, то есть по сути в картинку "
          "«частота на время». Кодировщик сворачивает её в последовательность "
          "представлений, декодировщик авторегрессивно порождает текст, "
          "попутно предсказывая язык, наличие речи и временные метки. "
          "Обучение шло на 680 тысячах часов разноязычного аудио, из-за чего "
          "модель устойчива к акцентам и шуму без настройки.")
text(doc, "Модуль существует не ради красивой расшифровки, а чтобы текст "
          "попал в модуль M1: нарушения в голосовых иначе не отлавливаются "
          "вообще.")

h3(doc, "Структура датасета")
text(doc, f"Размеченных публичных корпусов под домен «голосовые комментарии "
          f"к объявлениям» нет, поэтому набор синтезирован: 15 фраз на языке "
          f"объявлений озвучены системным голосом macOS, эталонный текст "
          f"известен по построению. Для приближения к реальным условиям "
          f"добавлен белый шум двух уровней, итого 45 записей. Фактическое "
          f"отношение сигнал/шум измерено по готовым файлам: около "
          f"{n(SNR['noisy15'], 0)} дБ для умеренного шума и {n(SNR['noisy10'], 0)} дБ "
          f"для сильного. Ограничение: синтезированная речь чище живой, "
          f"полученные значения ошибки распознавания — нижняя граница.")

h3(doc, "Метрики качества")
table(doc, ["Условие", "WER", "CER", "RTF", "Согласованность вердиктов"],
      [[COND[c] + (f", {n(SNR[c], 0)} дБ" if c in SNR else ""),
        n(C_SMALL[c]["wer"], 3), n(C_SMALL[c]["cer"], 3), n(C_SMALL[c]["rtf_mean"], 3),
        pc(C_SMALL[c]["verdict_agreement"])] for c in COND],
      widths=[1.8, 0.8, 0.8, 0.8, 2.1], size=10, align_right=(1, 2, 3, 4))
text(doc, f"Главная метрика здесь не ошибка распознавания, а согласованность "
          f"вердиктов — доля записей, где решение M1 по расшифровке совпало "
          f"с решением по эталонному тексту. Сравнение первых двух строк "
          f"показательно: при умеренном шуме ошибка распознавания выросла "
          f"в {n(C_SMALL['noisy15']['wer'] / C_SMALL['clean']['wer'], 1)} раза, "
          f"а согласованность не изменилась — модель путает окончания, но "
          f"оскорбление остаётся оскорблением. При сильном шуме согласованность "
          f"падает до {pc(C_SMALL['noisy10']['verdict_agreement'])}: опасны не "
          f"любые ошибки, а те, что уничтожают ключевое слово:")
code(doc, [
    f"эталон:      {LOST_KEYWORD['reference']}",
    f"распознано:  {LOST_KEYWORD['transcription']}",
    f"вердикт:     {LOST_KEYWORD['expected']} → {LOST_KEYWORD['got']}",
])
text(doc, "Оскорбление растворилось в шуме, и нарушение прошло бы "
          "в публикацию. Именно это и обнаруживает выбранная метрика — "
          "показатель WER такого не показывает.")
if CLEAN_FLIP:
    text(doc, f"Единственное расхождение на чистой записи другого рода. Фраза "
              f"«{CLEAN_FLIP['reference']}» распознана почти дословно, "
              f"с ошибками только в окончаниях, но вердикт сменился с "
              f"{CLEAN_FLIP['expected']} на {CLEAN_FLIP['got']}. Оценка M1 для "
              f"этого текста находится вблизи нижнего порога, и небольшого "
              f"изменения формулировки хватает, чтобы перейти через него. Это "
              f"следствие низких порогов M1 (раздел 5.1), а не ошибка "
              f"распознавания.")

h3(doc, "Особенности реализации и эффективность")
bullets(doc, [
    [("Квантизация int8 ", True, False),
     ("уменьшает размер весов и ускоряет вычисления на процессоре; сравнение "
      "с float32 в работе не проводилось, выбор сделан по рекомендации "
      "разработчиков faster-whisper.", False, False)],
    [("Фильтр речевой активности ", True, False),
     ("отсекает тишину до подачи в модель.", False, False)],
    [("Функция распознавания возвращает генератор, ", True, False),
     ("и вычисления происходят только при обходе. В коде это сделано явно — "
      "иначе замер времени показал бы микросекунды вместо реальной работы.",
      False, False)],
])
screenshot(doc, "scr_m3_cli", "Запуск M3: распознавание голосового комментария "
                              "и передача расшифровки в M1")

# ---------------------------------- M4 -------------------------------------- #
h2(doc, "5.4. M4 — видео: проверка видеообзора товара")

h3(doc, "Преимущества библиотеки")
text(doc, "Ultralytics берёт на себя чтение видео, масштабирование "
          "с сохранением пропорций, подавление дублирующих рамок и отрисовку. "
          "Параметр пропуска кадров реализован на уровне загрузчика — "
          "пропущенные кадры не декодируются вовсе. Потоковый режим отдаёт "
          "генератор результатов: видео обрабатывается покадрово и не грузится "
          "в память целиком.")

h3(doc, "Принцип работы модели и причины выбора")
text(doc, "YOLO11n (2,6 млн параметров, 5,4 МБ) — одностадийный детектор "
          "без якорных рамок. Изображение проходит через свёрточный "
          "экстрактор признаков, признаки с разных масштабов объединяются, "
          "а голова напрямую предсказывает для каждой точки сетки смещения "
          "рамки, уверенность и класс. В отличие от двухстадийных архитектур "
          "нет отдельного этапа генерации кандидатов — отсюда скорость. "
          "80 классов COCO хорошо ложатся на товарные категории: велосипед, "
          "автомобиль, диван, ноутбук, телевизор, холодильник, собака, кошка.")

h3(doc, "Структура датасета")
text(doc, f"Обучение: COCO, около 118 тысяч изображений, 80 классов, разметка "
          f"прямоугольниками. Инженерное следствие: из 23 категорий справочника "
          f"видеомодуль способен проверить 16, остальные помечены как "
          f"непроверяемые. Оценка: 12 роликов по {CLIP_SECONDS} секунд, собранных "
          f"из фотографий Imagenette медленным наездом с панорамированием — "
          f"для детектора это близко к тому, что снимает продавец, обходя "
          f"товар. На каждый ролик две проверки: с верной категорией "
          f"и с подменённой, итого {M4['n_checks']}. Ограничения: в таких роликах "
          f"нет смазов и ухода объекта из кадра, а подменная категория "
          f"назначается сдвигом по справочнику и визуально далека от истинной, "
          f"поэтому её обнаружение — простая часть задачи.")

h3(doc, "Метрики качества")
text(doc, f"Шаг между анализируемыми кадрами выбран по данным, а не на глаз: "
          f"при шаге {STRIDE} качество то же, что при полном переборе, а времени "
          f"тратится в {n(FULL4['elapsed_s'] / W4['elapsed_s'], 1)} раза меньше.")
table(doc, ["Шаг кадров", "Accuracy", "Подмены пойманы", "Время (GPU), с"],
      [[(f"**{c}**" if s == STRIDE else c) for c in
        (str(s), n(YN_MPS[s]["accuracy"], 3), n(YN_MPS[s]["mismatch_recall"], 3),
         n(YN_MPS[s]["elapsed_s"], 1))] for s in STRIDES],
      widths=[1.3, 1.4, 1.9, 2.0], size=10, align_right=(0, 1, 2, 3))
text(doc, f"Время указано за полный проход по {M4['n_checks']} проверкам.")
figure(doc, "m4_stride", IMG / "m4_stride_sweep.png", 5.2,
       "Зависимость качества и времени обработки от шага "
       "между анализируемыми кадрами")

PER_CAT = W4["per_category"]
table(doc, ["Категория", "Роликов", "Подтверждено"],
      [[c, str(v["videos"]), f"{v['confirmed']} / {v['videos']}"] for c, v in PER_CAT.items()],
      widths=[2.4, 1.2, 1.8], size=10, align_right=(1, 2))
worst = min(PER_CAT, key=lambda c: PER_CAT[c]["confirmed"] / PER_CAT[c]["videos"])
text(doc, f"Ошибки сосредоточены в категории «{worst}», и причина конкретна: "
          f"класс COCO «спортивный мяч» обучен на футбольных, теннисных "
          f"и баскетбольных мячах, а мяч для гольфа крупным планом детектор "
          f"принимает за «яблоко» и другие округлые предметы. Вывод для "
          f"эксплуатации: в каталог допустимо класть только категории, надёжно "
          f"покрытые классами обучающего набора.")
figure(doc, "m4_detection", IMG / "m4_detection.png", 5.2,
       "Кадр из выходного видео M4 с рамками детектора: ролик категории "
       "«Животные: собаки»")

h3(doc, "Особенности реализации и эффективность")
bullets(doc, [
    [("Фильтр единичных срабатываний: ", True, False),
     ("объект засчитывается только если найден минимум в трёх кадрах — "
      "одиночное срабатывание на одном кадре почти наверняка ошибка "
      "детектора. Отсеянные классы возвращаются отдельным полем, чтобы "
      "решение было объяснимо.", False, False)],
    [("Категория подтверждается набором классов, а не одним. ", True, False),
     ("При этом «Спецтехника» отделена от «Автомобилей»: пока они были слиты, "
      "мусоровоз подтверждался как легковой автомобиль, и подмены ловились "
      "в 91,7 % вместо нынешних 100 %.", False, False)],
    [("Скорость: ", True, False),
     (f"при шаге {STRIDE} ролик длительностью 30 секунд обрабатывается "
      f"примерно за {n(SEC_30_MPS, 1)} с на GPU и {n(SEC_30_CPU, 1)} с на "
      f"процессоре (пересчёт с {CLIP_SECONDS}-секундных роликов).", False, False)],
])
screenshot(doc, "scr_m4_cli", "Запуск M4: найденные объекты и подтверждение "
                              "заявленной категории")

# ---------------------------------- M5 -------------------------------------- #
h2(doc, "5.5. M5 — локальная языковая модель: вердикт и объяснение")

h3(doc, "Преимущества библиотеки")
text(doc, "Ollama — обёртка над llama.cpp, снимающая возню с квантизацией "
          "и выгрузкой слоёв на ускоритель: модель ставится одной командой "
          "и поднимает HTTP API на локальном адресе. Решающая возможность — "
          "структурированный вывод: в запрос передаётся JSON-схема, и сервер "
          "ограничивает генерацию так, что невалидный JSON получиться "
          "не может. Это не постобработка ответа, а ограничение на этапе "
          "выбора токенов — отсюда 100 % разбираемых ответов. Сервер также "
          "отдаёт собственные счётчики токенов, поэтому скорость берётся "
          "от него, а не оценивается клиентом.")
screenshot(doc, "scr_ollama", "Локальное развёртывание: модели qwen2.5 "
                              "установлены, 3B загружена и работает на GPU",
           width_in=4.5)

h3(doc, "Принцип работы модели и причины выбора")
text(doc, "Qwen2.5-Instruct в квантизации Q4_K_M — трансформер-декодировщик, "
          "порождающий текст по одному токену с обусловливанием на всё "
          "предыдущее. Версия Instruct дообучена следовать инструкциям, "
          "поэтому держит формат ответа и системный промпт. Квантизация "
          "означает хранение весов примерно в 4 битах на параметр вместо 16: "
          "модель на 3 млрд параметров умещается в 1,9 ГБ.")
rich(doc, [("Принципиальное ограничение роли: языковая модель ", False, False),
           ("не принимает решение", True, False),
           (". Вердикт вычисляет детерминированное правило — «побеждает самое "
            "строгое решение среди модулей». Модель получает готовые числовые "
            "результаты M1-M4 и делает то, что правилом не выражается: "
            "формулирует объяснение для продавца и сводку для модератора.",
            False, False)])

h3(doc, "Структура датасета")
text(doc, f"Обучение: многоязычный веб-корпус порядка 18 трлн токенов плюс "
          f"этап настройки на инструкциях. Инференс: агрегированный JSON "
          f"с решениями модулей M1-M4; контент повторно не подаётся. Оценка: "
          f"{N_CASES} синтетических кейсов, построенных систематическим перебором "
          f"сочетаний решений — всё чисто, по одному нарушению в каждом модуле, "
          f"конфликтующие сочетания, неполные объявления. Эталонный вердикт "
          f"вычисляется тем же правилом, что и в эксплуатации, а не "
          f"проставляется руками.")

h3(doc, "Метрики качества")
text(doc, f"Каждая модель прогнана по {N_CASES} кейсам {Q3['repeats']} раза — "
          f"{N5} запросов, чтобы отделить устойчивый результат от случайности "
          f"сэмплирования. Помимо вердикта, текст для продавца проверяется "
          f"автоматически на формальные дефекты: служебные термины системы "
          f"(allow, review, reject), слова из смешанных алфавитов, пустой или "
          f"слишком длинный ответ.")
table(doc, ["Показатель", "qwen2.5-7b", "qwen2.5-3b", "Критерий ТЗ"], [
    ["Валидный JSON с первой попытки", pc(Q7["valid_json_rate"], 0), pc(Q3["valid_json_rate"], 0),
     "не ниже 95 %"],
    ["Совпадение вердикта с правилом", pc(Q7["verdict_agreement"]),
     f"**{pc(Q3['verdict_agreement'])}**", "не ниже 90 %"],
    ["Текст для продавца без дефектов", f"**{pc(Q7['clean_text_rate'])}**",
     pc(Q3["clean_text_rate"]), "—"],
    ["Скорость генерации", f"{n(Q7['generation_tps']['mean'], 1)} ток/с",
     f"**{n(Q3['generation_tps']['mean'], 1)} ток/с**", "не ниже 15 ток/с"],
    ["Время ответа, среднее / p95",
     f"{n(Q7['response_s']['mean'])} / {n(Q7['response_s']['p95'])} с",
     f"**{n(Q3['response_s']['mean'])} / {n(Q3['response_s']['p95'])} с**", "—"],
    ["Память под модель", f"{Q7['memory']['model_mb']:.0f} МБ",
     f"**{Q3['memory']['model_mb']:.0f} МБ**", "не выше 12 ГБ"],
], widths=[2.1, 1.3, 1.3, 1.9], size=10, align_right=(1, 2))
text(doc, f"Модель на 7 млрд параметров ошиблась с вердиктом {Q7_ERR} раз из {N5}, "
          f"и все ошибки воспроизводятся во всех повторах на одних и тех же "
          f"кейсах: «{'», «'.join(Q7_ERR_CASES)}». В смешанных кейсах она видит "
          f"два разных решения и выбирает мягкое, хотя правило приоритета "
          f"задано в системном промпте явно; в кейсе с одним текстом "
          f"отсутствующие проверки она трактует как повод для ручной проверки. "
          f"Совпадение вердикта у неё к тому же нестабильно между прогонами: "
          f"при предыдущем прогоне того же набора было 87,5 %. Меньшая модель "
          f"применяет правило буквально и не ошиблась ни разу.")
if Q3_DEFECTS:
    text(doc, f"Зато по тексту для продавца картина обратная: у 3B "
              f"служебные термины найдены в {Q3_DEFECTS} ответах из {N5}, "
              f"например: «{Q3_DEFECT_EXAMPLE}» У 7B таких ответов "
              f"{'нет' if not Q7_DEFECTS else Q7_DEFECTS}. Выбор 3B сделан потому, что ошибка в тексте "
              f"исправима постобработкой (замена служебных терминов), а ошибка "
              f"смысла — нет; но это размен, а не превосходство по всем "
              f"показателям.")
text(doc, "Это подтверждает архитектурное решение: если бы вердикт выносила "
          "языковая модель, выбор размера стал бы критичным — неверный статус "
          "получало бы заметное число объявлений, систематически в сторону "
          "смягчения. Поскольку вердикт считает правило, ошибка модели "
          "на решение не влияет.")

h3(doc, "Особенности реализации и эффективность")
bullets(doc, [
    [("Схема вместо просьбы: ", True, False),
     ("формат задаётся JSON-схемой в параметре запроса, а не фразой «ответь "
      "в формате JSON». Отсюда 100 % валидности.", False, False)],
    [("Объяснение не противоречит вердикту: ", True, False),
     ("если вердикт модели разошёлся с правилом, объяснение запрашивается "
      "повторно с уже вычисленным вердиктом в промпте. Без этого продавец "
      "прочитал бы «объявление на проверке» при фактическом отказе.",
      False, False)],
    [("Низкая температура (0,2): ", True, False),
     ("задача — не творчество, а воспроизводимая формулировка.", False, False)],
    [("Память меряется через служебный эндпоинт сервера, ", True, False),
     ("а перед замером все модели выгружаются. Две итерации до правильного "
      "замера описаны в разделе 8.", False, False)],
])
screenshot(doc, "scr_m5_cli", "Запуск M5: результаты проверок M1–M4 на входе "
                              "и JSON-ответ локальной модели")

# ---------------------- 6. сравнение размеров моделей ----------------------- #
doc.add_page_break()
h1(doc, "6. Сравнение моделей разного размера")

text(doc, "В четырёх модулях замерялись модели разного размера, чтобы "
          "обоснование выбора опиралось на измерение, а не на цифры "
          "из карточки модели. В модуле изображений вместо этого сравнивались "
          "zero-shot и обученный классификатор (раздел 5.2).")

W = {m: M3[m]["conditions"] for m in W_SIZES}
table(doc, ["Модуль", "Модели", "Качество", "Скорость", "Вывод"], [
    ["M1\nтекст", "rubert-tiny 11,8M\nпротив\nrussian_toxicity 178M",
     f"публичный корпус:\n{n(M1_TINY_PUBLIC['roc_auc'], 4)} против "
     f"{n(M1_HEAVY_PUBLIC['roc_auc'], 4)}\nобъявления:\n"
     f"{n(M1_TINY_DOMAIN['roc_auc'], 4)} против **{n(M1['roc_auc'], 4)}**",
     f"{n(M1_LAT_TINY['latency_ms']['p50'])} мс\nпротив "
     f"{n(M1_LAT_HEAVY['latency_ms']['p50'])} мс",
     f"**Окупается.** На исходном домене модели неразличимы при "
     f"{SPEED_RATIO:.0f}-кратной разнице в скорости, а на новом разрыв "
     f"{ROC_GAP} пунктов: лёгкая не видит целых видов нарушений"],
    ["M3\nаудио", "whisper tiny / base / small\n39M / 74M / 244M",
     "согласованность вердиктов:\nчисто — " +
     " / ".join(pc(W[m]["clean"]["verdict_agreement"]) for m in W_SIZES) +
     "\nсильный шум — " +
     " / ".join(pc(W[m]["noisy10"]["verdict_agreement"]) for m in W_SIZES),
     "RTF " + " / ".join(n(W[m]["clean"]["rtf_mean"], 3) for m in W_SIZES),
     "**Не окупается на чистом звуке:** все три размера дают одинаковую "
     "согласованность при разной скорости. Окупается под шумом"],
    ["M4\nвидео", "yolo11n 2,6M\nпротив yolo11s 9,4M",
     f"при шаге {STRIDE}: {n(YN_MPS[STRIDE]['accuracy'], 3)} против "
     f"{n(YS_MPS[STRIDE]['accuracy'], 3)}\nпри шаге {MAX_STRIDE}: "
     f"{n(YN_MPS[MAX_STRIDE]['accuracy'], 3)} против "
     f"{n(YS_MPS[MAX_STRIDE]['accuracy'], 3)}",
     f"{n(YN_MPS[STRIDE]['elapsed_s'], 1)} с против {n(YS_MPS[STRIDE]['elapsed_s'], 1)} с",
     f"**Не окупается при рабочем шаге:** крупная тратит на "
     f"{(YS_MPS[STRIDE]['elapsed_s'] / YN_MPS[STRIDE]['elapsed_s'] - 1) * 100:.0f} % "
     f"больше времени при той же точности. Окупается при редком анализе кадров"],
    ["M5\nLLM", "qwen2.5 7B\nпротив qwen2.5 3B",
     f"вердикт: {pc(Q7['verdict_agreement'])} против **{pc(Q3['verdict_agreement'])}**\n"
     f"текст без дефектов: **{pc(Q7['clean_text_rate'])}** против "
     f"{pc(Q3['clean_text_rate'])}",
     f"{n(Q7['generation_tps']['mean'], 1)} против **{n(Q3['generation_tps']['mean'], 1)}** "
     f"ток/с\n{Q7['memory']['model_mb']:.0f} против **{Q3['memory']['model_mb']:.0f}** МБ",
     "**Не окупается для этой задачи:** меньшая модель точнее следует правилу "
     "и вдвое быстрее, крупная пишет чище. Задача требует буквального "
     "исполнения правила, дефекты текста исправимы постобработкой"],
], widths=[0.7, 1.6, 1.9, 1.2, 2.3], size=9)

text(doc, f"Отдельно измерен выигрыш от ускорителя. В модуле изображений "
          f"переход с процессора на GPU даёт {n(min(GPU_GAIN), 1)}–"
          f"{n(max(GPU_GAIN), 1)} раза, а батчинг — {n(BATCH_GAIN_MPS, 1)}–"
          f"{n(BATCH_GAIN_CPU, 1)} раза; в модуле видео GPU даёт "
          f"{n(min(GPU_GAIN_4), 1)}–{n(max(GPU_GAIN_4), 1)} раза, причём выигрыш "
          f"растёт с размером модели и падает с увеличением шага кадров.")

text(doc, "Общий вывод: крупная модель однозначно окупилась только в модуле "
          "текста — там решала полнота покрытия видов нарушений, которой "
          "у лёгкой модели не было. В аудио и видео крупная модель выигрывает "
          "лишь в тяжёлых условиях (сильный шум, редкий анализ кадров), "
          "а в LLM меньшая модель лучше подходит под задачу. Сравнивать модели "
          "надо в условиях эксплуатации и на данных своего домена.")

# ---------------------- 7. критерии приёмки --------------------------------- #
h1(doc, "7. Выполнение критериев приёмки")
text(doc, "Сводка по критериям из технического задания (SPEC.md). Для M1 "
          "приведены обе оценки — на данных подбора порогов и при "
          "кросс-валидации, — поскольку они расходятся.")
q3_ok = (Q3["valid_json_rate"] >= 0.95 and Q3["verdict_agreement"] >= 0.9
         and Q3["generation_tps"]["mean"] >= 15 and Q3["memory"]["model_mb"] <= 12 * 1024)
table(doc, ["Модуль", "Критерий", "Факт", "Статус"], [
    ["M1", "precision(reject) ≥ 0,95",
     f"{n(B1['precision_reject'])} на данных подбора; "
     f"{n(B_CV['precision_reject_at_prior'])} при кросс-валидации "
     f"в пересчёте на поток", "**не подтверждён** на отложенных данных"],
    ["M1", "доля review ≤ 20 %",
     f"{pc(B1['share_review'])}; при кросс-валидации {pc(B_CV['share_review'])}",
     status(B1["share_review"] <= 0.2 and B_CV["share_review"] <= 0.2)],
    ["M1", "p95 < 100 мс на CPU", f"{n(LAT_C['p95'], 1)} мс", status(LAT_C["p95"] < 100)],
    ["M2", "top-1 ≥ 0,85", n(CLIP_DIS["top1_accuracy"], 3),
     status(CLIP_DIS["top1_accuracy"] >= 0.85)],
    ["M2", "recall подмены ≥ 0,80 при precision ≥ 0,90",
     f"{n(MM['recall_mismatch'], 3)} при {n(MM['precision_reject'], 3)}",
     status(MM["recall_mismatch"] >= 0.8 and MM["precision_reject"] >= 0.9)],
    ["M2", "< 200 мс на изображение",
     f"{n(PERF['mps']['1']['ms_per_image'], 1)} мс GPU, "
     f"{n(PERF['cpu']['1']['ms_per_image'], 1)} мс CPU",
     status(PERF["cpu"]["1"]["ms_per_image"] < 200)],
    ["M3", "WER ≤ 0,20 при нормальном качестве", n(C_SMALL["clean"]["wer"], 3),
     status(C_SMALL["clean"]["wer"] <= 0.2)],
    ["M3", "согласованность вердиктов ≥ 0,90",
     " / ".join(f"{COND[c]} {pc(C_SMALL[c]['verdict_agreement'])}" for c in COND),
     "выполнен без шума и при умеренном шуме, **не выполнен** при сильном"],
    ["M3", "RTF ≤ 0,5", n(max(C_SMALL[c]["rtf_mean"] for c in COND), 2),
     status(max(C_SMALL[c]["rtf_mean"] for c in COND) <= 0.5)],
    ["M4", "accuracy ≥ 0,85", f"{n(W4['accuracy'], 3)} при шаге {STRIDE}",
     status(W4["accuracy"] >= 0.85)],
    ["M4", "ролик 30 с за ≤ 30 с", f"≈ {n(SEC_30_MPS, 1)} с GPU, ≈ {n(SEC_30_CPU, 1)} с CPU",
     status(SEC_30_CPU <= 30)],
    ["M5", "валидный JSON ≥ 95 %, вердикт ≥ 90 %, ≥ 15 ток/с, ≤ 12 ГБ",
     f"qwen2.5-3b: {pc(Q3['valid_json_rate'], 0)}, {pc(Q3['verdict_agreement'], 0)}, "
     f"{n(Q3['generation_tps']['mean'], 0)} ток/с, {n(Q3['memory']['model_mb'] / 1024, 1)} ГБ",
     status(q3_ok)],
], widths=[0.6, 2.0, 2.5, 1.5], size=9.5)
text(doc, "Не выполнены два критерия. Согласованность вердиктов в M3 при "
          "сильном шуме упирается в качество распознавания; ошибка здесь ведёт "
          "к пропуску нарушения, а не к ложной блокировке. Точность отказа "
          "в M1 упирается в объём размеченных объявлений: на отложенных данных "
          "появились ложные отказы честных объявлений, поэтому автоматический "
          "отказ в M1 рекомендуется включать только после калибровки "
          "на реальном потоке.")

# --------------------------- 8. проблемы ------------------------------------ #
doc.add_page_break()
h1(doc, "8. Возникшие проблемы и их решения")
text(doc, "Часть проблем — ошибки измерения, дававшие правдоподобные, "
          "но неверные цифры. Такие случаи опаснее явных сбоев: они "
          "не падают.")

h2(doc, "8.1. Инфраструктура и совместимость")
table(doc, ["Проблема", "Решение"], [
    ["Загрузка весов с Hugging Face зависала на нуле байт: процесс жив, "
     "файл не растёт",
     "Причина в транспортном бэкенде xet. Отключён переменной "
     "HF_HUB_DISABLE_XET=1, загрузка пошла по обычному HTTP"],
    ["Библиотека datasets версии 4 перестала поддерживать датасеты "
     "со скриптами загрузки",
     "Выбраны репозитории, хранящие данные в формате parquet; наличие "
     "parquet-файлов проверяется до скачивания"],
    ["Хранилище fast.ai недоступно из сети",
     "Тот же датасет взят из зеркала на Hugging Face"],
    ["В Transformers версии 5 метод получения эмбеддингов возвращает объект, "
     "а не тензор",
     "Нужное поле — pooler_output. Корректность проверена сверкой с выходом "
     "полного прохода модели: совпадение до седьмого знака"],
    ["Загрузка второй языковой модели для сравнения зависла на 4,1 ГБ из 4,9",
     "Вместо модели того же размера другого семейства взята модель меньшего "
     "размера того же семейства. Эксперимент стал полезнее: вопрос «окупается "
     "ли размер» информативнее вопроса «какой вендор лучше»"],
    ["Оболочка zsh не разбивает переменную на аргументы, и цикл запуска "
     "прогонов молча ничего не выполнял",
     "Команды запускаются явно, без подстановки строки аргументов"],
    ["Тестовые данные для изображений и видео не попадали в репозиторий, "
     "и на чистой копии оценка не воспроизводилась",
     "Добавлены скрипты загрузки Imagenette и подготовки кадров для роликов"],
], widths=[3.2, 3.4], size=10)

h2(doc, "8.2. Ошибки измерения")
table(doc, ["Проблема", "Решение"], [
    ["Общий модуль замеров сообщал устройство «GPU» для модулей, фактически "
     "работающих на процессоре: брал доступное в системе, а не используемое",
     "Фактическое устройство передаётся в замер явно; все затронутые "
     "измерения перепроведены"],
    ["Память под языковую модель измерялась по процессу с именем ollama "
     "и показывала 41 МБ вместо 4,5 ГБ",
     "Веса держит дочерний процесс llama-server, а не управляющий"],
    ["После исправления замер стал суммировать все такие процессы и приписал "
     "модели на 1,9 ГБ семь гигабайт",
     "Сервер держит модель в памяти ещё пять минут после запроса, поэтому "
     "при сравнении двух моделей в памяти были обе. Теперь перед замером все "
     "модели выгружаются, размер берётся из служебного эндпоинта"],
    ["Функция распознавания речи возвращает генератор: замер времени показывал "
     "бы микросекунды вместо реальной работы",
     "Результат явно материализуется внутри замеряемого участка"],
    ["Пороги подбирались на выборке с долей нарушений 17 % и 50 %, тогда как "
     "в эксплуатации ожидается 5 %",
     "Точность пересчитывается на ожидаемую долю аналитически через доли "
     "верных и ложных срабатываний — без выбрасывания размеченных данных"],
    ["Пороги текста подбирались и проверялись на одних и тех же 200 примерах, "
     "оценка была завышена",
     "Добавлена кросс-валидация; она показала, что требование к точности "
     "на новых данных не подтверждается (раздел 5.1)"],
    ["Метрика подмены категории для фото считала подмену пойманной при любом "
     "предсказании, кроме самой подменной категории, без порога уверенности",
     "Метрика считается той же функцией решения, что работает в эксплуатации, "
     "с разбивкой по трём корзинам и точностью отказа"],
    ["Уровни шума в аудио назывались по децибелам, но задавались амплитудой "
     "генератора",
     "Фактическое отношение сигнал/шум измерено по готовым файлам"],
], widths=[3.2, 3.4], size=10)

h2(doc, "8.3. Содержательные проблемы")
table(doc, ["Проблема", "Решение"], [
    ["У модулей фото и видео были собственные списки категорий, "
     "противоречившие друг другу",
     "Введён единый справочник. Побочный эффект: стало видно, что видеомодуль "
     "способен проверить только 16 категорий из 23"],
    ["Категории «Автомобили» и «Спецтехника» были слиты, мусоровоз "
     "подтверждался как легковой автомобиль",
     "Категории разделены по классам детектора; обнаружение подмены выросло "
     "с 91,7 % до 100 %"],
    ["Выбранная для текста модель оказалась слепа к угрозам и презрению. "
     "Обнаружено только после расширения тестового набора с 60 до 200 примеров "
     "с разбивкой по видам нарушений",
     "Решение передано модели с лучшим покрытием, разбивка по видам нарушений "
     "оставлена за лёгкой моделью — каскад с ленивой загрузкой второй"],
    [f"Ленивая загрузка второй модели в HTTP-сервисе давала "
     f"{n(LAT_C['max'] / 1000, 1)} с на первом запросе с нарушением",
     "В сервисе обе модели загружаются и прогреваются при старте"],
    ["Выбор максимума по видам нарушений всегда что-нибудь возвращал: голова "
     "«опасный контент» срабатывала даже на текстах, которые модель считает "
     "чистыми",
     "Введён порог уверенности для объяснения. Ниже него система сообщает, "
     "что вид нарушения не определён, вместо того чтобы назвать выдуманную "
     "причину"],
    ["Два требования ТЗ оказались несовместимы: «поймать 95 % нарушений» "
     "и «не более 20 % потока на ручную проверку»",
     "Построена кривая размена, рабочая точка выбрана по цене ошибки"],
    ["При расхождении вердикта LLM с правилом объяснение для продавца "
     "описывало бы не тот вердикт",
     "Объяснение запрашивается повторно с вычисленным правилом вердиктом"],
    ["Класс COCO «спортивный мяч» не покрывает мячи для гольфа",
     "Ограничение зафиксировано в справочнике и в отчёте; категории без "
     "надёжного покрытия помечены как непроверяемые"],
], widths=[3.2, 3.4], size=10)

# ------------------------------ 9. выводы ----------------------------------- #
doc.add_page_break()
h1(doc, "9. Выводы")

h2(doc, "9.1. Что подтвердилось на практике")
bullets(doc, [
    [("Готовые модели решают прикладные задачи без обучения. ", True, False),
     ("Пять содержательных задач закрыты предобученными весами: от детектора "
      "на 5 МБ до языковой модели на 3 млрд параметров.", False, False)],
    [("Тестовый набор определяет, что вы увидите. ", True, False),
     (f"Пока доменный набор для текста состоял из 60 примеров «обычные "
      f"объявления против грубых», лёгкая модель показывала ROC-AUC 0,95 "
      f"и выглядела достаточной. Как только добавились угрозы без брани, "
      f"презрение и законные резкие формулировки, её показатель упал до "
      f"{n(M1_TINY_DOMAIN['roc_auc'], 2)}, а у тяжёлой остался "
      f"{n(M1['roc_auc'], 2)}. Модели не изменились — изменился набор.",
      False, False)],
    [("Оценка на данных подбора завышена. ", True, False),
     (f"Пороги текста, подобранные на 200 примерах, давали точность отказа "
      f"{n(B1['precision_reject'])}; кросс-валидация показала "
      f"{n(B_CV['precision_reject_at_prior'], 2)} в пересчёте на поток. "
      f"Для порогов в области сотых долей двухсот примеров мало.", False, False)],
    [("Модель детектирует то, что видела, а не то, что названо в её метках. ",
      True, False),
     ("У лёгкой модели есть голова «угроза», но на фразе «найду и проблемы "
      "будут» она выдаёт ноль. Наличие метки в конфигурации ничего "
      "не гарантирует.", False, False)],
    [("Правильная метрика важнее хорошей модели. ", True, False),
     (f"В модуле аудио рост ошибки распознавания в "
      f"{n(C_SMALL['noisy15']['wer'] / C_SMALL['clean']['wer'], 1)} раза при "
      f"умеренном шуме не снизил согласованность вердиктов, зато при сильном шуме "
      f"отдельные ошибки меняли вердикт на противоположный. Измеряй мы только "
      f"ошибку распознавания — этого не было бы видно.", False, False)],
    [("Крупная модель окупается не всегда. ", True, False),
     ("Однозначно окупилась только в модуле текста; в аудио и видео — лишь "
      "в тяжёлых условиях, в LLM меньшая модель лучше подошла под задачу "
      "(раздел 6).", False, False)],
    [("Инженерная оптимизация может стоить дороже железа. ", True, False),
     (f"Батчинг ускоряет в {n(BATCH_GAIN_MPS, 1)}–{n(BATCH_GAIN_CPU, 1)} раза, "
      f"а переход на GPU — в {n(min(GPU_GAIN), 1)}–{n(max(GPU_GAIN), 1)}. "
      f"Требовать ускоритель там, где достаточно накопить пачку запросов, — "
      f"дорогая ошибка.", False, False)],
    [("Требования могут оказаться несовместимыми, и это нормальный результат. ",
      True, False),
     ("Правильный ответ — не подкручивать метрику, а предъявить кривую размена "
      "и выбрать точку по цене ошибки.", False, False)],
])

h2(doc, "9.2. Ограничения работы")
table(doc, ["Что", "В чём ограничение"], [
    ["Текст, публичный корпус",
     "модели обучались на данных того же происхождения — оценка сверху"],
    ["Текст, доменный набор",
     "200 примеров, составлены автором: стилистика однороднее реальной"],
    ["Текст, пороги",
     f"при кросс-валидации точность отказа {n(B_CV['precision_reject_at_prior'], 2)} "
     f"в пересчёте на поток вместо требуемых 0,95; нужна калибровка "
     f"на реальных объявлениях"],
    ["Изображения", "подменная категория визуально далека от истинной; "
                    "подмена на близкую категорию не измерялась"],
    ["Аудио", "синтезированная речь чище живой, ошибка занижена; "
              "int8 не сравнивался с float32"],
    ["Видео", "ролики собраны из фотографий трёх классов; 7 категорий "
              "из 23 не проверяются"],
    ["Языковая модель", f"{N_CASES} синтетических кейсов; качество текста "
                        f"проверялось только формальными правилами"],
], widths=[2.1, 4.5], size=10)

h2(doc, "9.3. Направления доработки")
bullets(doc, [
    [("Собрать доменный набор для текста из реальных объявлений "
      "с естественной долей нарушений и перекалибровать пороги.", False, False)],
    [("Записать живые голосовые сообщения и видеообзоры вместо "
      "синтезированных.", False, False)],
    [("Добавить в оценку фото и видео подмену на близкую категорию.",
      False, False)],
    [("Проверить whisper-medium и yolo11m: окупается ли рост качества "
      "замедлением.", False, False)],
    [("Сравнить Ollama с MLX — нативный для Apple Silicon рантайм.",
      False, False)],
])

# --------------------------- 10. флоу разработки ---------------------------- #
doc.add_page_break()
h1(doc, "10. Процесс разработки с ИИ-агентом")

text(doc, "Работа выполнялась в режиме агентной разработки. Ниже описаны "
          "инструменты, модель, пайплайн и расход ресурсов.")

h2(doc, "10.1. Сетап")
table(doc, ["Компонент", "Что использовалось"], [
    ["Агент", "Claude Code, десктопное приложение"],
    ["Модель", "Claude Opus 5"],
    ["Инструменты агента",
     "чтение и правка файлов, запуск команд оболочки, фоновые задачи "
     "для долгих загрузок, панель браузера для проверки документов, "
     "скилл генерации .docx"],
    ["Среда", "macOS 26.6, Apple M4 Pro, Python 3.12 в изолированном окружении"],
    ["Контроль версий", "git, репозиторий создавался вручную"],
], widths=[1.7, 4.9], size=10)

h2(doc, "10.2. Расход ресурсов")
table(doc, ["Показатель", "Значение"], [
    ["Размер контекста сессии", "445 065 токенов, 45 % окна в 1 млн"],
    ["Из них переписка", "381 161 токен (86 %)"],
    ["Определения инструментов", "52 846 токенов"],
    ["Расход тарифного плана", "17 % пятичасового лимита, 3 % недельного "
                               "(план Pro)"],
], widths=[2.6, 4.0], size=10)
text(doc, "Важная оговорка: 445 тысяч токенов — это размер контекста на конец "
          "работы, а не суммарно оплаченные токены. Кумулятивный расход выше, "
          "поскольку контекст пересылается на каждом шаге; реальное "
          "потребление отражают проценты тарифного плана.")

h2(doc, "10.3. Пайплайн")
text(doc, "Работа шла итерациями «сформулировать — реализовать — измерить — "
          "пересмотреть», а не одним проходом. Характерные шаги:")
bullets(doc, [
    [("Постановка. ", True, False),
     ("Первая версия плана начиналась с выбора моделей; это было отвергнуто "
      "и переделано в правильном порядке: прикладная задача, затем ТЗ "
      "с критериями, затем модели и только потом метрики.", False, False)],
    [("Реализация модуля. ", True, False),
     ("Сначала скрипт инференса с проверкой на одном примере, затем скрипт "
      "оценки с сохранением метрик в JSON. Долгие загрузки моделей "
      "и датасетов выносились в фоновые задачи, чтобы писать код параллельно.",
      False, False)],
    [("Проверка. ", True, False),
     ("Каждый модуль прогонялся на реальных данных сразу после написания. "
      "Это и выявляло проблемы из раздела 8 — например, расхождение API "
      "в новой версии библиотеки обнаружилось на первом же запуске.",
      False, False)],
    [("Пересмотр. ", True, False),
     ("Расширение тестового набора для текста изменило вывод по модулю "
      "и потребовало переписать его реализацию на каскад из двух моделей. "
      "Отдельное ревью готовой работы нашло завышенную оценку порогов "
      "и упрощённую метрику подмены — после исправления выводы по M1 и M5 "
      "стали осторожнее. Агентная разработка даёт скорость, но проверять "
      "выводы всё равно приходится данными.", False, False)],
    [("Отчётность. ", True, False),
     ("Отчёт генерируется скриптом, который читает измеренные значения "
      "из сохранённых метрик, поэтому после перезапуска оценки он "
      "пересобирается без ручной правки.", False, False)],
])

h2(doc, "10.4. Что оказалось трудным")
text(doc, "Все технические трудности перечислены в разделе 8, и большинство "
          "из них — именно трудности агентной работы. Отдельно стоит "
          "выделить два класса:")
bullets(doc, [
    [("Молчаливые ошибки измерения. ", True, False),
     ("Замер памяти языковой модели дважды подряд выдавал правдоподобные, "
      "но неверные цифры — сначала в сто раз меньше, затем втрое больше. "
      "Ни один из вариантов не вызывал ошибку, и заметить это можно было "
      "только сверкой с ожидаемым порядком величины.", False, False)],
    [("Смена API в свежих версиях библиотек. ", True, False),
     ("Два случая: изменившийся тип возвращаемого значения и удалённая "
      "поддержка части датасетов. Оба решались чтением фактического поведения, "
      "а не документации.", False, False)],
])

# --------------------------- 11. репозиторий -------------------------------- #
h1(doc, "11. Репозиторий")
para(doc, REPO_URL, size=12, bold=True)
if "<" in REPO_URL:
    para(doc, "(ссылку подставить в REPO_URL в report/make_docx.py после публикации)",
         size=10, italic=True, color=WARN, space_after=10)

h2(doc, "11.1. Состав")
code(doc, [
    "SPEC.md                техническое задание с критериями приёмки",
    "README.md              обзор проекта и инструкции запуска",
    "catalog.json           единый справочник категорий для M2 и M4",
    "requirements.txt       прямые зависимости с зафиксированными версиями",
    "requirements.lock.txt  полный слепок окружения (92 пакета)",
    "common/bench.py        единый модуль замеров для всех модулей",
    "m1_text/               текст:  predict / evaluate / api / bench_api",
    "m2_image/              фото:   predict / evaluate / download_data",
    "m3_audio/              аудио:  transcribe / evaluate / make_testset",
    "m4_video/              видео:  detect_video / evaluate / make_testset",
    "m5_llm/                LLM:    verdict / benchmark / make_cases",
    "report/metrics/        машинные результаты всех прогонов (JSON)",
    "report/img/            графики, схемы и скриншоты",
    "report/docx/           этот отчёт (.docx и .pdf)",
    "report/make_docx.py    сборка отчёта из измеренных метрик",
])

h2(doc, "11.2. Установка и запуск")
code(doc, [
    "brew install python@3.12 ffmpeg ollama",
    "python3.12 -m venv .venv",
    ".venv/bin/pip install -r requirements.txt",
    "ollama serve                     # в отдельном терминале",
    "ollama pull qwen2.5:3b-instruct",
    "",
    "# текст: инференс, метрики, HTTP-сервис",
    ".venv/bin/python m1_text/predict.py --text \"Продам гараж, документы готовы\"",
    ".venv/bin/python m1_text/evaluate.py --dataset m1_text/data/samples.csv \\",
    "    --prior 0.05 --target-recall 0.85 --cv 5 --tag domain_final",
    ".venv/bin/uvicorn m1_text.api:app --port 8000",
    "",
    "# данные для фото и видео",
    ".venv/bin/python m2_image/download_data.py",
    ".venv/bin/python m4_video/prepare_frames.py",
    ".venv/bin/python m4_video/make_testset.py",
    "",
    "# изображение, аудио, видео, языковая модель",
    ".venv/bin/python m2_image/evaluate.py --limit 500 --perf-devices mps cpu",
    ".venv/bin/python m3_audio/evaluate.py --sizes tiny base small",
    ".venv/bin/python m4_video/evaluate.py --models yolo11n.pt yolo11s.pt \\",
    "    --devices mps cpu --strides 1 5 10 15",
    ".venv/bin/python m5_llm/benchmark.py \\",
    "    --models qwen2.5:7b-instruct qwen2.5:3b-instruct --repeats 3",
    "",
    "# отчёт",
    ".venv/bin/python report/make_diagram.py",
    ".venv/bin/python report/make_docx.py",
])

h2(doc, "11.3. Открытость стека")
table(doc, ["Компонент", "Лицензия"], [
    ["Hugging Face Transformers, PyTorch, Ultralytics", "Apache 2.0 / BSD / AGPL"],
    ["faster-whisper, CTranslate2, Ollama, llama.cpp", "MIT"],
    ["Модели: rubert-tiny-toxicity, russian_toxicity_classifier, CLIP, Whisper, "
     "YOLO11, Qwen2.5", "открытые веса"],
    ["Датасеты: публичный корпус русских комментариев, Imagenette",
     "открытые наборы данных"],
], widths=[4.2, 2.4], size=10)
text(doc, "Платные подписки и облачные API не использовались, все вычисления "
          "выполняются локально.")

doc.save(str(OUT))
print("документ →", OUT)
missing = [SCREENSHOTS[f][0] for f in SCREENSHOTS if not (SCREENS / SCREENSHOTS[f][0]).exists()]
if missing:
    print(f"скриншотов не хватает ({len(missing)}): {', '.join(missing)}")
