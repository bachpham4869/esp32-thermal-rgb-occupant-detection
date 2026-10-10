#!/usr/bin/env python3
"""
eval_extra.py - Danh gia mo rong cho bai bao (bo sung cho train_classifier.py).

Tinh: holdout + bootstrap CI, LOSO (leave-one-session-out), so sanh nhieu model,
rule-only vs rule+LR, ablation feature, quet nguong, mo phong canh bao theo
cua so 1 giay, va ve hinh cho bai bao.

Output:
    docs/results/metrics.json   - tat ca so lieu
    docs/figures/*.png          - hinh

Chay:
    pip install numpy pandas scikit-learn matplotlib
    python3 tools/eval_extra.py
"""
import contextlib
import io
import json
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import train_classifier as tc  # noqa: E402

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve,
                             average_precision_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
F = tc.FEATURE_COLS
STATIC = ["aspect", "solidity", "deltaAmbT", "maxTemp", "tempSpread"]
TEMPORAL = ["trendSlope", "residStd", "cxJitter", "areaJitter"]
CLASS_ORDER = ["label1_human", "label2_hotcup", "label3_phone"]
CLASS_NAME = {"label1_human": "Human", "label2_hotcup": "Hot cup", "label3_phone": "Phone"}
# Bang mau tham chieu (3 slot dau validate all-pairs): blue / orange / aqua
COLORS = {"label1_human": "#2a78d6", "label2_hotcup": "#eb6834", "label3_phone": "#1baf7a"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

OUT_DIR = "docs/results"
FIG_DIR = "docs/figures"


def lr():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))


MODELS = {
    "LR (9 feat, deployed)": lr,
    "LR maxTemp only": lr,
    "LR static only (5)": lr,
    "LR temporal only (4)": lr,
    "Decision tree (d=4)": lambda: DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=SEED),
    "Random forest (100)": lambda: RandomForestClassifier(100, class_weight="balanced", random_state=SEED, n_jobs=-1),
    "MLP (16)": lambda: make_pipeline(StandardScaler(), MLPClassifier((16,), max_iter=2000, random_state=SEED)),
}
MODEL_FEATS = {
    "LR maxTemp only": ["maxTemp"],
    "LR static only (5)": STATIC,
    "LR temporal only (4)": TEMPORAL,
}


def feats(name):
    return MODEL_FEATS.get(name, F)


def metrics(y, yp, proba=None):
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    m = dict(n=int(len(y)), acc=accuracy_score(y, yp),
             precision=precision_score(y, yp, zero_division=0),
             recall=recall_score(y, yp, zero_division=0),
             f1=f1_score(y, yp, zero_division=0),
             specificity=tn / (tn + fp) if tn + fp else float("nan"),
             far=fp / (tn + fp) if tn + fp else float("nan"),
             cm=[[int(tn), int(fp)], [int(fn), int(tp)]])
    if proba is not None and len(np.unique(y)) > 1:
        m["auc"] = roc_auc_score(y, proba)
        m["ap"] = average_precision_score(y, proba)
    return m


def bootstrap_ci(y, yp, fn, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        vals.append(fn(y[s], yp[s]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def session_of(fname):
    # all_seat_blanket_occluded.csv (1 dong) va _2 la cung kich ban -> gop 1 session
    return re.sub(r"(_\d+)?\.csv$", "", fname)


def load():
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        df = tc.clean(tc.load_all_csv("data/raw"))
    df["session"] = df["source_file"].map(session_of)
    df["ts"] = pd.to_datetime(df["hostTimestamp"], errors="coerce")
    return df.reset_index(drop=True)


def rule_pass(df):
    # Rule cascade cua classify() ap lai tren feature da log; gray-zone gia dinh qua
    # (khong co variance 8-frame trong CSV) -> can tren cua recall cua rule.
    return ((df.maxTemp <= 58) & (df.pixelCount >= 15) &
            df.aspect.between(0.35, 2.8)).astype(int).values


def deployed_proba(df):
    m = json.load(open("data/models/lr_model.json"))
    z = df[m["feature_order"]].values @ np.array(m["weights"]) + m["bias"]
    return 1 / (1 + np.exp(-z))


def loso_predict(df, make, cols):
    proba = np.zeros(len(df))
    for s in df.session.unique():
        te = df.session == s
        tr = ~te
        if df.loc[tr, "is_human"].nunique() < 2:
            continue
        mdl = make().fit(df.loc[tr, cols], df.loc[tr, "is_human"])
        proba[te.values] = mdl.predict_proba(df.loc[te, cols])[:, 1]
    return proba


def window_alerts(df, proba, thr=0.5):
    """Mo phong muc he thong: gom theo cua so 1 giay (hostTimestamp), cua so = 'phat hien'
    neu co it nhat 1 blob qua rule va pHuman >= thr. Human session -> ti le giay phat
    hien; negative session -> ti le giay bao dong gia."""
    d = df[["session", "label_dir", "is_human", "ts"]].copy()
    d["hit"] = ((proba >= thr) & (rule_pass(df) == 1)).astype(int)
    d = d.dropna(subset=["ts"])
    d["sec"] = d.ts.dt.floor("s")
    w = d.groupby(["session", "label_dir", "is_human", "sec"]).hit.max().reset_index()
    out = {}
    for s, g in w.groupby("session"):
        g = g.sort_values("sec")
        hits = g.hit.values
        # debounce kieu N/M tren cua so giay: >=5/7 giay lien tiep -> alert
        alert = np.array([hits[max(0, i - 6):i + 1].sum() >= 5 for i in range(len(hits))])
        first = int(np.argmax(alert)) if alert.any() else None
        out[s] = dict(cls=g.label_dir.iloc[0], seconds=int(len(g)),
                      window_hit_rate=float(hits.mean()),
                      alert_rate=float(alert.mean()),
                      seconds_to_first_alert=first)
    return out


def threshold_sweep(y, proba):
    rows = []
    for t in np.round(np.arange(0.05, 1.0, 0.05), 2):
        yp = (proba >= t).astype(int)
        m = metrics(y, yp)
        rows.append(dict(thr=float(t), acc=m["acc"], precision=m["precision"],
                         recall=m["recall"], f1=m["f1"], far=m["far"]))
    return rows


# ------------------------------------------------------------------ figures
def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def make_figures(df, res, holdout_proba, y_hold, loso_proba):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "font.family": "DejaVu Sans"})
    os.makedirs(FIG_DIR, exist_ok=True)

    # Fig A: feature distributions (4 features x 3 classes)
    fs = ["maxTemp", "tempSpread", "cxJitter", "residStd"]
    units = {"maxTemp": "°C", "tempSpread": "°C", "cxJitter": "px", "residStd": "°C"}
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, f in zip(axes, fs):
        data = [df.loc[df.label_dir == c, f].values for c in CLASS_ORDER]
        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, lw=1.2),
                        whiskerprops=dict(color=INK2, lw=0.8), capprops=dict(color=INK2, lw=0.8))
        for patch, c in zip(bp["boxes"], CLASS_ORDER):
            patch.set_facecolor(COLORS[c]); patch.set_edgecolor("white"); patch.set_linewidth(2)
        ax.set_xticks([1, 2, 3], [CLASS_NAME[c] for c in CLASS_ORDER])
        ax.set_title(f"{f} ({units[f]})")
        style(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_feature_dist.png", dpi=200)
    plt.close(fig)

    # Fig B: ROC + PR (holdout vs LOSO) - 2 panel, cung 1 truc moi panel
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    y_all = df.is_human.values
    for (yy, pp, lab, col) in [(y_hold, holdout_proba, "Random 75/25 split", "#2a78d6"),
                               (y_all, loso_proba, "Leave-one-session-out", "#eb6834")]:
        fpr, tpr, _ = roc_curve(yy, pp)
        axes[0].plot(fpr, tpr, lw=2, color=col, label=f"{lab} (AUC {roc_auc_score(yy, pp):.3f})")
        pr, rc, _ = precision_recall_curve(yy, pp)
        axes[1].plot(rc, pr, lw=2, color=col, label=f"{lab} (AP {average_precision_score(yy, pp):.3f})")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color=INK2)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate"); axes[0].set_title("ROC")
    axes[1].set_xlabel("Recall (human)"); axes[1].set_ylabel("Precision (human)"); axes[1].set_title("Precision-recall")
    for ax in axes:
        style(ax); ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_roc_pr.png", dpi=200)
    plt.close(fig)

    # Fig C: LOSO accuracy per session (bar ngang, mau theo class)
    rows = res["loso_per_session"]
    names = list(rows.keys())
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    ypos = np.arange(len(names))[::-1]
    for yv, s in zip(ypos, names):
        r = rows[s]
        ax.barh(yv, r["acc"], height=0.6, color=COLORS[r["cls"]], edgecolor="white", lw=2)
        ax.text(r["acc"] + 0.01, yv, f"{r['acc']:.3f}  (n={r['n']})", va="center", fontsize=7, color=INK)
    ax.set_yticks(ypos, names, fontsize=7)
    ax.set_xlim(0, 1.18); ax.set_xlabel("Accuracy on held-out session")
    ax.set_title("Leave-one-session-out accuracy (LR, 9 features)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS[c], label=CLASS_NAME[c]) for c in CLASS_ORDER],
              fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_loso.png", dpi=200)
    plt.close(fig)

    # Fig D: rule vs rule+LR pass rate per class (grouped bar)
    pr = res["rules_vs_lr"]["pass_rate"]
    fig, ax = plt.subplots(figsize=(5, 2.8))
    x = np.arange(len(CLASS_ORDER))
    w = 0.36
    for k, (key, col, lab) in enumerate([("rules", "#86b6ef", "Rules only"),
                                          ("rules_lr", "#2a78d6", "Rules + LR gate")]):
        vals = [pr[c][key] * 100 for c in CLASS_ORDER]
        ax.bar(x + (k - 0.5) * w, vals, w, color=col, edgecolor="white", lw=2, label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (k - 0.5) * w, v + 2, f"{v:.1f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, [CLASS_NAME[c] for c in CLASS_ORDER])
    ax.set_ylabel("% samples accepted as human"); ax.set_ylim(0, 125)
    ax.set_title("Distractor rejection by the LR gate")
    ax.legend(fontsize=7, frameon=False, loc="upper center", ncol=2)
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_rules_vs_lr.png", dpi=200)
    plt.close(fig)

    # Fig E: standardized coefficients
    coef = res["standardized_coefficients"]
    order = sorted(coef, key=lambda k: abs(coef[k]))
    fig, ax = plt.subplots(figsize=(5, 3))
    vals = [coef[k] for k in order]
    ax.barh(order, vals, height=0.6, color=["#2a78d6" if v > 0 else "#eb6834" for v in vals],
            edgecolor="white", lw=2)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Standardized LR coefficient  (+ → human, − → distractor)")
    ax.set_title("Feature importance")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_coef.png", dpi=200)
    plt.close(fig)

    # Fig F: threshold sweep (LOSO) - cac dai luong cung thang 0..1, 1 truc
    sw = pd.DataFrame(res["threshold_sweep_loso"])
    fig, ax = plt.subplots(figsize=(5, 3))
    for key, col, lab in [("recall", "#2a78d6", "Recall (human)"),
                          ("precision", "#1baf7a", "Precision (human)"),
                          ("far", "#eb6834", "False-acceptance rate")]:
        ax.plot(sw.thr, sw[key], lw=2, color=col, label=lab)
    ax.axvline(0.5, color=INK2, lw=0.8, ls="--")
    ax.text(0.51, 0.5, "deployed\nthreshold", fontsize=7, color=INK2)
    ax.set_xlabel("Decision threshold on p(human)"); ax.set_ylim(0, 1.02)
    ax.set_title("Threshold sweep (LOSO predictions)")
    ax.legend(fontsize=7, frameon=False, loc="center left")
    style(ax)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fig_threshold.png", dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load()
    y = df.is_human.values
    res = {}

    # ---- dataset
    res["dataset"] = dict(
        rows=int(len(df)), positives=int(y.sum()), negatives=int((1 - y).sum()),
        per_session={s: int(n) for s, n in df.groupby("session").size().items()},
        per_class={c: int(n) for c, n in df.groupby("label_dir").size().items()},
        duration_s={s: float((g.ts.max() - g.ts.min()).total_seconds())
                    for s, g in df.groupby("session")},
        blobtype_crosstab=pd.crosstab(df.label_dir, df.blobType).to_dict(),
        feature_mean={c: g[F + ["pixelCount"]].mean().round(4).to_dict() for c, g in df.groupby("label_dir")},
        feature_std={c: g[F + ["pixelCount"]].std().round(4).to_dict() for c, g in df.groupby("label_dir")},
        feature_median={c: g[F + ["pixelCount"]].median().round(4).to_dict() for c, g in df.groupby("label_dir")},
    )

    # ---- holdout (giong train_classifier.py)
    Xtr, Xte, ytr, yte, itr, ite = train_test_split(df[F], y, np.arange(len(df)), test_size=0.25,
                                                    random_state=SEED, stratify=y)
    m = lr().fit(Xtr, ytr)
    p_hold = m.predict_proba(Xte)[:, 1]
    yp_hold = (p_hold >= 0.5).astype(int)
    res["holdout"] = metrics(yte, yp_hold, p_hold)
    res["holdout"]["acc_ci95"] = bootstrap_ci(yte, yp_hold, accuracy_score)
    res["holdout"]["f1_ci95"] = bootstrap_ci(yte, yp_hold, f1_score)
    res["holdout"]["per_class_pass"] = {
        c: float(yp_hold[df.label_dir.values[ite] == c].mean()) for c in CLASS_ORDER}

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    res["cv5_auc"] = float(cross_val_score(lr(), df[F], y, cv=cv, scoring="roc_auc").mean())
    res["cv5_acc"] = float(cross_val_score(lr(), df[F], y, cv=cv, scoring="accuracy").mean())

    # ---- LOSO
    p_loso = loso_predict(df, lr, F)
    yp_loso = (p_loso >= 0.5).astype(int)
    res["loso"] = metrics(y, yp_loso, p_loso)
    res["loso"]["acc_ci95"] = bootstrap_ci(y, yp_loso, accuracy_score)
    res["loso_per_session"] = {
        s: dict(cls=df.loc[df.session == s, "label_dir"].iloc[0], n=int((df.session == s).sum()),
                acc=float(accuracy_score(y[df.session == s], yp_loso[(df.session == s).values])),
                mean_p=float(p_loso[(df.session == s).values].mean()))
        for s in df.session.unique()}

    # ---- so sanh model (holdout + LOSO)
    comp = {}
    for name, make in MODELS.items():
        cols = feats(name)
        mm = make().fit(Xtr[cols], ytr)
        ph = mm.predict_proba(Xte[cols])[:, 1]
        pl = loso_predict(df, make, cols)
        comp[name] = dict(n_features=len(cols),
                          holdout=metrics(yte, (ph >= 0.5).astype(int), ph),
                          loso=metrics(y, (pl >= 0.5).astype(int), pl),
                          loso_cup_far=float((pl[df.label_dir == "label2_hotcup"] >= 0.5).mean()),
                          loso_phone_far=float((pl[df.label_dir == "label3_phone"] >= 0.5).mean()),
                          loso_blanket_recall=float((pl[df.session == "all_seat_blanket_occluded"] >= 0.5).mean()))
    res["model_comparison"] = comp

    # ---- rules vs rules+LR (LR deployed, in-sample) va rules + LR LOSO (out-of-sample)
    rp = rule_pass(df)
    p_dep = deployed_proba(df)
    lr_dep = (p_dep >= 0.5).astype(int)
    lr_loso = yp_loso
    res["rules_vs_lr"] = dict(
        rules=metrics(y, rp),
        rules_lr_deployed=metrics(y, rp & lr_dep),
        rules_lr_loso=metrics(y, rp & lr_loso),
        pass_rate={c: dict(rules=float(rp[df.label_dir == c].mean()),
                           rules_lr=float((rp & lr_dep)[df.label_dir == c].mean()),
                           rules_lr_loso=float((rp & lr_loso)[df.label_dir == c].mean()))
                   for c in CLASS_ORDER},
        rule_reject_reason={c: dict(too_hot=int((g.maxTemp > 58).sum()),
                                    too_small=int((g.pixelCount < 15).sum()),
                                    bad_aspect=int((~g.aspect.between(0.35, 2.8)).sum()))
                            for c, g in df.groupby("label_dir")},
    )

    # ---- ablation
    base = cross_val_score(lr(), df[F], y, cv=cv, scoring="roc_auc").mean()
    abl = {"all": float(base)}
    for c in F:
        abl[f"-{c}"] = float(cross_val_score(lr(), df[[x for x in F if x != c]], y, cv=cv, scoring="roc_auc").mean())
    abl_loso = {}
    for c in F:
        cols = [x for x in F if x != c]
        pl = loso_predict(df, lr, cols)
        abl_loso[f"-{c}"] = float(accuracy_score(y, (pl >= 0.5).astype(int)))
    res["ablation_cv5_auc"] = abl
    res["ablation_loso_acc"] = abl_loso

    # ---- he so chuan hoa
    full = lr().fit(df[F], y)
    res["standardized_coefficients"] = dict(zip(F, map(float, full[-1].coef_[0])))

    # ---- threshold sweep
    res["threshold_sweep_loso"] = threshold_sweep(y, p_loso)

    # ---- muc he thong (cua so 1 s + debounce 5/7)
    res["window_alerts_loso"] = window_alerts(df, p_loso)
    res["window_alerts_rules_only"] = window_alerts(df, np.ones(len(df)))

    # ---- ngan sach tinh toan LR
    res["lr_inference_cost"] = dict(params=len(F) + 1, macs=len(F), expf=1, bytes=4 * (len(F) + 1))

    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(res, f, indent=2, default=float)
    make_figures(df, res, p_hold, yte, p_loso)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
