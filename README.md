# Предмодерация объявлений с помощью готовых ML-моделей

Учебный проект: пять независимых модулей, каждый обрабатывает свой тип контента
и решает одну прикладную задачу — автоматическую предмодерацию объявлений
на площадке частных объявлений.

Модели используются **готовые, без обучения и дообучения**. Всё работает
**локально**, без обращения к внешним API.

- **[Отчёт (PDF)](report/docx/Отчёт_ML_предмодерация_объявлений.pdf)** — разбор
  моделей, метрик и принятых решений
- **[SPEC.md](SPEC.md)** — техническое задание с критериями приёмки
- **[catalog.json](catalog.json)** — справочник категорий, общий для M2 и M4

---

## Схема

```
объявление
  ├── текст ──────────► M1  детекция нарушений в тексте
  ├── фото ───────────► M2  соответствие фото заявленной категории
  ├── голосовое ──────► M3  распознавание речи ──► M1
  ├── видео ──────────► M4  проверка наличия заявленного объекта
  └── результаты M1-M4 ► M5  вердикт и формулировка причины (LLM)
```

Каждое решение попадает в одну из трёх корзин: `allow` (публиковать),
`review` (на ручную проверку), `reject` (отклонить).

**Почему три, а не две.** Ложная блокировка честного объявления обходится
площадке дороже, чем пропущенное нарушение: продавец теряет деньги и уходит,
тогда как пропуск ловится жалобами и постмодерацией. Поэтому в `reject`
попадает только то, в чём модель уверена, а всё пограничное уходит человеку.

---

## Модули и модели

| | Задача | Библиотека | Модель | Почему она |
|---|---|---|---|---|
| **M1** | текст | Hugging Face Transformers | каскад `russian_toxicity_classifier` + `rubert-tiny-toxicity` | тяжёлая решает (покрывает угрозы и презрение, которых лёгкая не видит), лёгкая даёт вид нарушения для объяснения |
| **M2** | изображение | Hugging Face Transformers | `openai/clip-vit-base-patch32` | zero-shot: категорий на площадке сотни и они меняются; новая категория добавляется строкой в JSON, а не переобучением |
| **M3** | аудио | faster-whisper (CTranslate2) | `whisper-small`, int8 | русский из коробки; int8 на CPU даёт RTF ≈ 0.27, без GPU |
| **M4** | видео | Ultralytics | `yolo11n` | 80 классов COCO хорошо покрывают товарные категории; 5.4 МБ, десятки FPS |
| **M5** | LLM | Ollama (llama.cpp) | `qwen2.5:7b-instruct` / `:3b-instruct`, Q4 | структурированный вывод по JSON-схеме; замер показал, что 3B справляется лучше 7B |

Использовано четыре разных фреймворка: Transformers, CTranslate2, Ultralytics
(поверх PyTorch) и llama.cpp.

---

## Установка

```bash
brew install python@3.12 ffmpeg ollama
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Для M5 дополнительно:

```bash
ollama serve        # в отдельном терминале
ollama pull qwen2.5:7b-instruct
ollama pull qwen2.5:3b-instruct     # для сравнения размеров
```

Веса моделей скачиваются автоматически при первом запуске и в репозиторий
не коммитятся. Тестовые данные M2 и M4 скачиваются и собираются скриптами:

```bash
.venv/bin/python m2_image/download_data.py      # Imagenette, ~460 МБ
.venv/bin/python m4_video/prepare_frames.py     # 12 кадров из Imagenette
.venv/bin/python m4_video/make_testset.py       # ролики из кадров
```

---

## Запуск

**M1 — текст**

```bash
.venv/bin/python m1_text/predict.py --text "Продам гараж, документы готовы"
.venv/bin/python m1_text/predict.py --text "..." --explain      # обе ступени каскада
.venv/bin/python m1_text/download_data.py --limit 6000          # публичный корпус
.venv/bin/python m1_text/evaluate.py --dataset m1_text/data/samples.csv \
    --prior 0.05 --target-recall 0.85 --cv 5 --tag domain_final \
    --sweep-recall 0.95 0.85 0.75 0.65 --save-thresholds        # калибровка
.venv/bin/uvicorn m1_text.api:app --port 8000                   # HTTP API
```

**M2 — изображение**

```bash
.venv/bin/python m2_image/predict.py --image фото.jpg --category "Спорт: мячи"
.venv/bin/python m2_image/evaluate.py --limit 500 --batch 16 \
    --perf-devices mps cpu
```

**M3 — аудио**

```bash
.venv/bin/python m3_audio/make_testset.py        # синтез эталонного набора
.venv/bin/python m3_audio/transcribe.py --audio запись.wav --moderate --srt
.venv/bin/python m3_audio/evaluate.py --sizes tiny base small
.venv/bin/python m3_audio/measure_snr.py         # фактический SNR шумных записей
```

**M4 — видео**

```bash
.venv/bin/python m4_video/make_testset.py
.venv/bin/python m4_video/detect_video.py --video обзор.mp4 --category "Велосипеды"
.venv/bin/python m4_video/evaluate.py --models yolo11n.pt yolo11s.pt \
    --devices mps cpu --strides 1 5 10 15
```

**M5 — LLM**

```bash
.venv/bin/python m5_llm/make_cases.py
.venv/bin/python m5_llm/verdict.py --index 0
.venv/bin/python m5_llm/benchmark.py \
    --models qwen2.5:7b-instruct qwen2.5:3b-instruct --repeats 3
```

---

## Метрики

Все замеры сохраняются в `report/metrics/*.json`, графики — в `report/img/`.
Отчёт собирается скриптом `report/make_docx.py`: числа в таблицах и тексте
читаются из этих файлов, поэтому после перезапуска оценки достаточно
пересобрать отчёт:

```bash
.venv/bin/python report/make_docx.py
soffice --headless --convert-to pdf --outdir report/docx \
    report/docx/Отчёт_ML_предмодерация_объявлений.docx
```

Единый модуль [`common/bench.py`](common/bench.py) меряет латентность (p50/p95),
RSS, память GPU и пропускную способность одинаково во всех модулях — иначе
цифры разных модулей нельзя сравнивать между собой.

---

## Структура

```
SPEC.md              техническое задание
catalog.json         единый справочник категорий (M2 и M4)
common/bench.py      общие замеры производительности
m1_text/             текст:      predict / evaluate / api / download_data
m2_image/            изображение: predict / evaluate / download_data
m3_audio/            аудио:      transcribe / evaluate / make_testset / measure_snr
m4_video/            видео:      detect_video / evaluate / make_testset / prepare_frames
m5_llm/              LLM:        verdict / benchmark / make_cases
report/metrics/      машинные результаты всех прогонов
report/img/          графики и схемы
report/docx/         итоговый отчёт (.docx и .pdf)
report/img/screens/  скриншоты запуска модулей для отчёта
report/make_docx.py  сборка отчёта из измеренных метрик
```
