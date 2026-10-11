#!/usr/bin/env python3
"""M1 — оценка качества и подбор рабочих порогов.

Пороги не назначаются произвольно, а выводятся из бизнес-требования ТЗ:
  * ложное отклонение честного объявления дорого  -> precision(reject) >= 0.95;
  * пропуск нарушения дёшев, но не бесплатен      -> в allow должно попадать
    не более 5% нарушений, остальное уходит на ручную проверку.

Результат: thresholds.json (используется predict.py и api.py)
           report/metrics/m1_*.json + графики в report/img/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.bench import Bench  # noqa: E402
from predict import MODEL_ID, TextModerator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "report" / "img"

TARGET_PRECISION = 0.95  # для порога reject
TARGET_RECALL = 0.95  # для порога review (сколько нарушений обязаны поймать)
MAX_REVIEW_SHARE = 0.20  # предел нагрузки на модераторов из ТЗ


def pick_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    prior: float | None = None,
    target_recall: float = TARGET_RECALL,
) -> dict:
    """Подбор порогов по требованиям ТЗ.

    precision зависит от доли нарушений в потоке, а размеченные наборы почти
    всегда сбалансированы искусственно. Пересэмплировать набор под реальную
    долю расточительно: чтобы из 100/100 получить 5%, пришлось бы выбросить
    95 нарушений из 100.

    Поэтому precision пересчитывается аналитически из TPR и FPR, которые
    от баланса классов не зависят:

        precision(p) = p·TPR / (p·TPR + (1−p)·FPR)

    где p — ожидаемая доля нарушений в проде. Так используются все
    размеченные данные, а порог целится в рабочие условия.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y_true, scores)
    observed = float(y_true.mean())
    p = observed if prior is None else prior

    denom = p * tpr + (1 - p) * fpr
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(denom > 0, p * tpr / denom, 0.0)

    # reject: минимальный порог, при котором precision достигает цели.
    # Среди подходящих берём тот, что даёт наибольший recall.
    ok = np.where((precision >= TARGET_PRECISION) & (tpr > 0))[0]
    if len(ok):
        best = ok[np.argmax(tpr[ok])]
        t_reject = float(thr[best])
        reached_precision = float(precision[best])
        reached_recall = float(tpr[best])
    else:
        best = int(np.argmax(precision))
        t_reject = float(thr[best])
        reached_precision = float(precision[best])
        reached_recall = float(tpr[best])

    # review: максимальный порог, при котором recall ещё не ниже цели
    ok = np.where(tpr >= target_recall)[0]
    t_review = float(thr[ok[0]]) if len(ok) else 0.5

    # roc_curve первым элементом отдаёт порог inf — он непригоден
    t_reject = min(t_reject, 1.0)
    t_review = min(t_review, t_reject)

    return {
        "review": round(t_review, 4),
        "reject": round(t_reject, 4),
        "target_precision": TARGET_PRECISION,
        "target_recall": target_recall,
        "assumed_violation_rate": round(p, 4),
        "observed_violation_rate": round(observed, 4),
        "expected_precision_at_rate": round(reached_precision, 4),
        "expected_recall_at_rate": round(reached_recall, 4),
    }


def bucket_stats(y_true: np.ndarray, scores: np.ndarray, th: dict) -> dict:
    """Продуктовые метрики: как поток делится на три корзины."""
    reject = scores >= th["reject"]
    review = (scores >= th["review"]) & ~reject
    allow = ~reject & ~review
    n = len(y_true)

    def safe(x, y):
        return round(float(x / y), 4) if y else 0.0

    return {
        "share_allow": safe(allow.sum(), n),
        "share_review": safe(review.sum(), n),
        "share_reject": safe(reject.sum(), n),
        "automated_share": safe(allow.sum() + reject.sum(), n),
        "precision_reject": safe((y_true[reject] == 1).sum(), reject.sum()),
        "recall_reject": safe((y_true[reject] == 1).sum(), (y_true == 1).sum()),
        # главный риск: нарушения, ушедшие в автопубликацию
        "violations_leaked_to_allow": safe((y_true[allow] == 1).sum(), (y_true == 1).sum()),
        "clean_wrongly_rejected": safe((y_true[reject] == 0).sum(), (y_true == 0).sum()),
    }


def by_kind(df, scores, th: dict) -> dict:
    """Разбивка по подгруппам набора.

    Средний показатель скрывает самое важное: какие виды нарушений модель
    ловит, а какие пропускает, и не срабатывает ли она на законных, но резких
    формулировках («перекупщиков не беспокоить»). Для калибровки порога это
    существеннее общей accuracy.
    """
    import numpy as np

    out = {}
    for kind, group in df.groupby("kind"):
        idx = group.index.to_numpy()
        sc = scores[df.index.get_indexer(idx)]
        label = int(group["label"].iloc[0])
        flagged = (sc >= th["reject"]).sum()
        to_review = ((sc >= th["review"]) & (sc < th["reject"])).sum()
        out[kind] = {
            "label": label,
            "n": len(group),
            "median_score": round(float(np.median(sc)), 4),
            "reject": int(flagged),
            "review": int(to_review),
            "allow": int(len(group) - flagged - to_review),
            # для нарушений это recall, для чистых — доля ложных отклонений
            "share_rejected": round(float(flagged / len(group)), 4),
        }
    return out


def cross_validate(df, y_true: np.ndarray, scores: np.ndarray, k: int,
                   prior: float | None, target_recall: float) -> dict:
    """Честная оценка порогов: подбор и проверка на разных примерах.

    Если пороги подобраны на тех же 200 текстах, на которых потом считаются
    метрики, результат завышен: порог подстраивается под конкретные примеры.
    Здесь набор делится на k частей; пороги подбираются на k−1 частях,
    а решение для каждого текста принимается порогами, которые этот текст
    не видели. Разбиение стратифицировано по подгруппам, чтобы в каждой
    части были все виды нарушений.
    """
    from sklearn.model_selection import StratifiedKFold

    strata = df["kind"].to_numpy() if "kind" in df.columns else y_true
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    oof_bucket = np.zeros(len(y_true), dtype=int)  # 0 allow, 1 review, 2 reject
    folds = []
    for train, test in skf.split(scores, strata):
        th = pick_thresholds(y_true[train], scores[train], prior, target_recall)
        sc = scores[test]
        oof_bucket[test] = np.select([sc >= th["reject"], sc >= th["review"]],
                                     [2, 1], default=0)
        folds.append({"review": th["review"], "reject": th["reject"]})

    # bucket_stats работает с порогами, поэтому out-of-fold корзины
    # подаются как «оценки» 0/1/2 с порогами 1 и 2
    buckets = bucket_stats(y_true, oof_bucket.astype(float), {"review": 1, "reject": 2})
    kinds = {}
    if "kind" in df.columns:
        for kind, idx in df.groupby("kind").indices.items():
            b = oof_bucket[idx]
            kinds[kind] = {
                "label": int(y_true[idx[0]]),
                "n": len(idx),
                "reject": int((b == 2).sum()),
                "review": int((b == 1).sum()),
                "allow": int((b == 0).sum()),
            }
    # та же поправка на базовую частоту, что и при подборе порогов
    p = float(y_true.mean()) if prior is None else prior
    tpr, fpr = buckets["recall_reject"], buckets["clean_wrongly_rejected"]
    denom = p * tpr + (1 - p) * fpr
    buckets["precision_reject_at_prior"] = round(p * tpr / denom, 4) if denom else 0.0
    buckets["prior"] = p
    rev = [f["review"] for f in folds]
    rej = [f["reject"] for f in folds]
    return {
        "k": k,
        "buckets": buckets,
        "by_kind": kinds,
        "fold_thresholds": folds,
        "threshold_range": {
            "review": [round(min(rev), 4), round(max(rev), 4)],
            "reject": [round(min(rej), 4), round(max(rej), 4)],
        },
    }


def plot_curves(y_true, scores, th, model_name: str, tag: str = "") -> None:
    """Три панели, каждая отвечает на свой вопрос.

    1. ROC — насколько вообще разделимы классы и где на этой кривой стоят
       выбранные пороги.
    2. Precision-Recall в двух вариантах: как на размеченной выборке
       и как будет в потоке с реальной долей нарушений. Расхождение между
       кривыми и есть эффект базовой частоты класса.
    3. Распределение по трём корзинам. Обычная матрица ошибок здесь не годится:
       она бинарна и схлопывает корзину ручной проверки, которой в системе
       отведена отдельная роль.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    prior = th.get("assumed_violation_rate", float(np.mean(y_true)))
    observed = float(np.mean(y_true))

    fpr, tpr, thr = roc_curve(y_true, scores)

    def at(threshold):
        """FPR и TPR в точке заданного порога."""
        i = int(np.argmin(np.abs(thr - threshold)))
        return fpr[i], tpr[i]

    # холст намеренно небольшой: в отчёте рисунок сжимается до ширины полосы
    # набора, и при большом холсте подписи становятся нечитаемыми
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))

    # --- 1. ROC ---------------------------------------------------------- #
    ax[0].plot(fpr, tpr, lw=1.8)
    ax[0].plot([0, 1], [0, 1], "--", c="gray", lw=0.8)
    for name, t, color in [("порог review", th["review"], "tab:orange"),
                           ("порог reject", th["reject"], "tab:red")]:
        x, y = at(t)
        ax[0].plot(x, y, "o", color=color, ms=9,
                   label=f"{name} = {t:g}")
    ax[0].set(xlabel="FPR — ложные\nсрабатывания",
              ylabel="TPR — пойманные\nнарушения", title="ROC-кривая")
    ax[0].legend(fontsize=10, loc="lower right")
    ax[0].grid(alpha=0.25)

    # --- 2. Precision-Recall при двух базовых частотах -------------------- #
    def precision_at(p):
        denom = p * tpr + (1 - p) * fpr
        return np.where(denom > 0, p * tpr / denom, 1.0)

    ax[1].plot(tpr, precision_at(observed), lw=1.8,
               label=f"выборка: {observed:.0%} наруш.")
    ax[1].plot(tpr, precision_at(prior), lw=2.2, color="tab:green",
               label=f"поток: {prior:.0%} наруш.")
    ax[1].axhline(TARGET_PRECISION, ls="--", c="r", lw=1.1,
                  label=f"требование {TARGET_PRECISION}")
    xr, yr = at(th["reject"])
    pr_at_reject = prior * yr / (prior * yr + (1 - prior) * xr) if (prior * yr + (1 - prior) * xr) else 1.0
    ax[1].plot(yr, pr_at_reject, "o", color="tab:red", ms=9,
               label=f"рабочая точка")
    ax[1].set(xlabel="Recall — пойманные\nнарушения", ylabel="Precision",
              title="Точность против полноты", ylim=(0.4, 1.03))
    ax[1].legend(fontsize=9.5, loc="lower left")
    ax[1].grid(alpha=0.25)

    # --- 3. Три корзины --------------------------------------------------- #
    scores = np.asarray(scores)
    y_true = np.asarray(y_true)
    buckets = np.select(
        [scores >= th["reject"], scores >= th["review"]], [2, 1], default=0
    )
    counts = np.zeros((2, 3), dtype=int)
    for cls in (0, 1):
        for b in (0, 1, 2):
            counts[cls, b] = int(((y_true == cls) & (buckets == b)).sum())

    ax[2].imshow(counts / counts.sum(axis=1, keepdims=True),
                 cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax[2].set_xticks(range(3), ["allow", "review", "reject"], fontsize=11)
    ax[2].set_yticks(range(2), ["чистые", "наруш."], fontsize=11)
    for cls in (0, 1):
        for b in (0, 1, 2):
            share = counts[cls, b] / counts[cls].sum()
            ax[2].text(b, cls, f"{counts[cls, b]}\n{share:.0%}", ha="center",
                       va="center", fontsize=13, weight="bold",
                       color="white" if share > 0.5 else "black")
    ax[2].set_title(f"Корзины при порогах\n{th['review']:g} и {th['reject']:g}",
                    fontsize=12)

    fig.suptitle(f"Модуль M1, {model_name.split('/')[-1]}, "
                 f"набор объявлений n={len(y_true)}", fontsize=12.5)
    fig.tight_layout()
    out = IMG_DIR / f"m1_{model_name.split('/')[-1]}{('_' + tag) if tag else ''}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("графики →", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="CSV с колонками text,label")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--limit", type=int, default=0, help="ограничить размер выборки")
    ap.add_argument("--positive-rate", type=float, default=0.0,
                    help="физически пересэмплировать выборку к заданной доле "
                         "нарушений; 0 — оставить как есть")
    ap.add_argument("--prior", type=float, default=0.0,
                    help="ожидаемая доля нарушений в проде (например 0.05). "
                         "precision пересчитывается на неё аналитически, "
                         "без выбрасывания данных — предпочтительнее "
                         "--positive-rate")
    ap.add_argument("--tag", default="", help="суффикс к имени файла метрик")
    ap.add_argument("--target-recall", type=float, default=TARGET_RECALL,
                    help="какую долю нарушений обязаны поймать (review+reject). "
                         "Снижение разгружает модераторов ценой утечки")
    ap.add_argument("--sweep-recall", type=float, nargs="*", default=[],
                    help="перебрать несколько значений и показать компромисс "
                         "«нагрузка на модераторов против утечки»")
    ap.add_argument("--cv", type=int, default=0,
                    help="k-кратная кросс-валидация порогов: подбор на k−1 частях, "
                         "проверка на оставшейся. 0 — не проводить")
    ap.add_argument("--save-thresholds", action="store_true")
    ap.add_argument("--use-saved-thresholds", action="store_true",
                    help="не подбирать пороги заново, а применить сохранённые — "
                         "так проверяется перенос на другой домен")
    args = ap.parse_args()

    import pandas as pd
    from sklearn.metrics import average_precision_score, classification_report, roc_auc_score

    df = pd.read_csv(args.dataset)
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=42)

    original_rate = float(df["label"].mean())
    if args.positive_rate:
        # precision зависит от базовой частоты класса: на выборке 50/50 порог,
        # дающий precision 0.95, в потоке с 5% нарушений даст заметно меньше.
        # Поэтому доля приводится к ожидаемой в проде ДО подбора порогов.
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        need_neg = int(len(pos) * (1 - args.positive_rate) / args.positive_rate)
        if need_neg <= len(neg):
            neg = neg.sample(need_neg, random_state=42)
        else:
            # негативов не хватает — уменьшаем число позитивов
            pos = pos.sample(
                int(len(neg) * args.positive_rate / (1 - args.positive_rate)),
                random_state=42,
            )
        df = pd.concat([pos, neg]).sample(frac=1, random_state=42)
        print(f"доля нарушений приведена: {original_rate:.1%} -> {df['label'].mean():.1%}")
    texts = df["text"].astype(str).tolist()
    y_true = df["label"].astype(int).to_numpy()
    print(f"выборка: {len(texts)} | нарушений: {y_true.sum()} ({y_true.mean():.1%})")

    mod = TextModerator(args.model)
    with Bench(f"m1_eval_{args.model.split('/')[-1]}", device=mod.device,
               extra={"model": args.model, "batch_size": 32}) as b:
        scores = []
        for i in range(0, len(texts), 32):
            chunk = texts[i : i + 32]
            with b.step():
                scores.extend(mod.toxicity(s) for s in mod.scores(chunk))
    scores = np.asarray(scores)

    if args.use_saved_thresholds:
        from predict import load_thresholds

        th = load_thresholds()
        print("\nприменяются сохранённые пороги:", json.dumps(th, ensure_ascii=False))
    else:
        th = pick_thresholds(y_true, scores, args.prior or None, args.target_recall)
        print("\nподобранные пороги:", json.dumps(th, ensure_ascii=False, indent=2))

    y_pred = (scores >= th["reject"]).astype(int)
    print("\n" + classification_report(y_true, y_pred,
                                       target_names=["чистый", "нарушение"], digits=3))

    metrics = {
        "model": args.model,
        "dataset": args.dataset,
        "n_samples": len(texts),
        "original_positive_rate": round(original_rate, 4),
        "positive_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_true, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_true, scores)), 4),
        "thresholds": th,
        "buckets": bucket_stats(y_true, scores, th),
        "performance": b.summary(),
    }
    print("\nROC-AUC:", metrics["roc_auc"], "| PR-AUC:", metrics["pr_auc"])
    print("корзины:", json.dumps(metrics["buckets"], ensure_ascii=False, indent=2))

    if args.sweep_recall:
        print(f"\n{'target_recall':>13} {'review порог':>13} {'reject порог':>13} "
              f"{'нагрузка':>9} {'утечка':>8} {'ложн.откл':>10}")
        sweep = []
        for tr in args.sweep_recall:
            t = pick_thresholds(y_true, scores, args.prior or None, tr)
            b = bucket_stats(y_true, scores, t)
            flag = "" if b["share_review"] <= MAX_REVIEW_SHARE else "  (сверх ТЗ)"
            print(f"{tr:>13.2f} {t['review']:>13.4f} {t['reject']:>13.4f} "
                  f"{b['share_review']:>9.3f} {b['violations_leaked_to_allow']:>8.3f} "
                  f"{b['clean_wrongly_rejected']:>10.3f}{flag}")
            sweep.append({"target_recall": tr, "thresholds": t, "buckets": b})
        metrics["recall_sweep"] = sweep

    kinds = {}
    if "kind" in df.columns:
        kinds = by_kind(df.reset_index(drop=True), scores, th)
        print(f"\n{'подгруппа':12s} {'класс':>6s} {'n':>4s} {'медиана':>8s} "
              f"{'reject':>7s} {'review':>7s} {'allow':>6s}")
        for name, k in sorted(kinds.items(), key=lambda x: (x[1]["label"], x[0])):
            cls = "наруш." if k["label"] else "чистый"
            print(f"{name:12s} {cls:>6s} {k['n']:>4d} {k['median_score']:>8.4f} "
                  f"{k['reject']:>7d} {k['review']:>7d} {k['allow']:>6d}")
        metrics["by_kind"] = kinds

    if args.cv:
        cv = cross_validate(df.reset_index(drop=True), y_true, scores, args.cv,
                            args.prior or None, args.target_recall)
        b = cv["buckets"]
        print(f"\nкросс-валидация, {args.cv} частей (пороги не видели проверяемых текстов):")
        print(f"  precision(reject) {b['precision_reject']:.3f} | recall(reject) "
              f"{b['recall_reject']:.3f} | review {b['share_review']:.1%} | "
              f"утечка {b['violations_leaked_to_allow']:.1%} | "
              f"ложные отклонения {b['clean_wrongly_rejected']:.1%}")
        print(f"  precision(reject) в потоке с {b['prior']:.0%} нарушений: "
              f"{b['precision_reject_at_prior']:.3f}")
        print(f"  разброс порогов по частям: {cv['threshold_range']}")
        metrics["cv"] = cv

    plot_curves(y_true, scores, th, args.model, args.tag)

    tag = f"_{args.tag}" if args.tag else ""
    out = ROOT / "report" / "metrics" / f"m1_{args.model.split('/')[-1]}{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("метрики →", out)

    if args.save_thresholds:
        # пороги привязаны к модели и к данным, на которых подобраны, —
        # без этой пометки невозможно понять, применимы ли они к другому домену
        th_out = {
            **th,
            "fitted_on": {
                "dataset": args.dataset,
                "model": args.model,
                "n_samples": len(texts),
                "positive_rate": round(float(y_true.mean()), 4),
                "date": __import__("datetime").date.today().isoformat(),
            },
            "_warning": "Пороги не переносятся между доменами. Перед эксплуатацией "
                        "перекалибруйте на размеченных данных целевого домена: "
                        "пороги, подобранные на форумных комментариях, на "
                        "объявлениях пропускают большую часть нарушений "
                        "(см. report/metrics/m1_*_domain.json).",
        }
        p = Path(__file__).parent / "thresholds.json"
        p.write_text(json.dumps(th_out, ensure_ascii=False, indent=2), encoding="utf-8")
        print("пороги →", p)


if __name__ == "__main__":
    main()
