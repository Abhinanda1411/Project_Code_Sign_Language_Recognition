"""
make_poster_figures.py - turns YOUR real results into poster-ready figures.

Usage (from the src folder):
    python make_poster_figures.py          # real data  -> poster_figures/
    python make_poster_figures.py --demo   # layout test with FAKE data (watermarked)

Sizes: every figure is drawn at its PRINTED size in inches, so fonts are real
points when you place the picture at 100 % on the poster
(axis labels/ticks 24-26 pt, table text 22-24 pt).

Input files (a missing file is skipped with a message, nothing is invented):
  data/raw/session_NN/<CLASS>.csv     -> Fig 1 (dataset size)
  results/model_comparison.csv        model,val_accuracy,val_macro_f1,inference_ms
  results/test_predictions.csv        y_true,y_pred   (final model, test session only)
  results/live_trials.csv             letter,trials,correct
  results/nlp_eval.csv                condition,keystrokes
"""
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

DEMO = "--demo" in sys.argv
OUT = os.path.join(C.PROJECT_ROOT, "poster_figures_demo" if DEMO else "poster_figures")
RES = os.path.join(C.PROJECT_ROOT, "results")
os.makedirs(OUT, exist_ok=True)

AX, TICK, TABLE, CAP = 26, 24, 24, 22
BLUE, GREEN, ORANGE, GREY = "#2F6DB5", "#2E8B63", "#C9822B", "#6b7580"

plt.rcParams.update({"font.size": TICK, "axes.labelsize": AX, "xtick.labelsize": TICK,
                     "ytick.labelsize": TICK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 2})


def finish(fig, name, caption):
    """Caption goes UNDER the figure (22 pt), demo watermark if fake data."""
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=CAP, wrap=True)
    if DEMO:
        fig.text(0.5, 0.5, "SAMPLE DATA - NOT REAL", ha="center", va="center",
                 fontsize=60, color="red", alpha=0.25, rotation=25)
    fig.savefig(os.path.join(OUT, name), dpi=150, facecolor="white")
    plt.close(fig)
    print("saved", name)


def need(path):
    if os.path.exists(path):
        return True
    print("skipped (file not found yet):", os.path.relpath(path, C.PROJECT_ROOT))
    return False


# ---------------- demo data (only with --demo) ----------------
def demo_frames():
    rng = np.random.default_rng(1)
    classes = C.ALL_CLASSES
    models = ["KNN", "SVM", "Random Forest", "MLP"]
    mc = pd.DataFrame({"model": models, "val_accuracy": [.90, .95, .93, .96],
                       "val_macro_f1": [.89, .95, .92, .96], "inference_ms": [2.1, 1.2, 4.0, 0.9]})
    yt = rng.choice(classes, 1400)
    yp = np.where(rng.random(1400) < .94, yt, rng.choice(classes, 1400))
    tp = pd.DataFrame({"y_true": yt, "y_pred": yp})
    lt = pd.DataFrame({"letter": classes, "trials": 10, "correct": rng.integers(7, 11, len(classes))})
    nl = pd.DataFrame({"condition": ["Letter by letter", "With word prediction"], "keystrokes": [100, 62]})
    return mc, tp, lt, nl


# ---------------- Fig 1: dataset size ----------------
def fig_dataset():
    rows = []
    if DEMO:
        for s in range(1, C.NUM_SESSIONS + 1):
            for h in C.HANDS:
                rows.append((s, h, 25 * 25 + 90))
    else:
        for f in glob.glob(os.path.join(C.RAW_DIR, "session_*", "*.csv")):
            df = pd.read_csv(f, usecols=lambda c: c in ("sample_id", "hand", "session", "label"))
            if "sample_id" in df.columns:                      # dynamic: many rows per sample
                df = df.drop_duplicates(["session", "hand", "sample_id"])
            for (s, h), n in df.groupby(["session", "hand"]).size().items():
                rows.append((int(s), h, int(n)))
    if not rows:
        print("skipped Fig 1: no recorded data found"); return
    d = pd.DataFrame(rows, columns=["session", "hand", "n"]).groupby(["session", "hand"]).n.sum().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    bottom = np.zeros(len(d))
    for h, col in zip(d.columns, (BLUE, ORANGE)):
        ax.bar(d.index, d[h], bottom=bottom, color=col, label=f"{h} hand", width=.7)
        bottom += d[h].values
    ax.set_xticks(d.index)
    ax.set_xticklabels([f"S{s}\n{C.SESSION_SPLIT.get(s, '')}" for s in d.index])
    ax.set_ylabel("Recorded samples"); ax.legend(fontsize=TICK, frameon=False)
    fig.subplots_adjust(bottom=.27, left=.14)
    finish(fig, "fig1_dataset_size.png", "Figure 1: Samples recorded in each session (train / validation / test).")


# ---------------- Fig 2: model comparison + timing table ----------------
def fig_models(mc):
    fig, (ax, tx) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1.3]})
    x = np.arange(len(mc)); w = .38
    ax.bar(x - w / 2, mc.val_accuracy * 100, w, color=BLUE, label="Accuracy")
    ax.bar(x + w / 2, mc.val_macro_f1 * 100, w, color=GREEN, label="Macro F1")
    ax.set_xticks(x); ax.set_xticklabels(mc.model)
    ax.set_ylabel("Validation score (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=TICK, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    tx.axis("off")
    t = tx.table(cellText=[[m, f"{ms:.1f} ms"] for m, ms in zip(mc.model, mc.inference_ms)],
                 colLabels=["Model", "Time per prediction"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(TABLE); t.scale(1, 2.6)
    fig.subplots_adjust(top=.9, bottom=.1, hspace=.3, left=.12)
    finish(fig, "fig2_model_comparison.png", "Figure 2: Models compared on the validation session, with prediction time.")


# ---------------- Fig 3: confusion matrix (test session) ----------------
def fig_confusion(tp):
    labels = [c for c in C.ALL_CLASSES if c in set(tp.y_true) | set(tp.y_pred)]
    m = pd.crosstab(tp.y_true, tp.y_pred).reindex(index=labels, columns=labels, fill_value=0)
    m = m.div(m.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=TICK, rotation=90)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=TICK)
    ax.set_xlabel("Predicted letter"); ax.set_ylabel("Actual letter")
    cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.02); cb.ax.tick_params(labelsize=TICK)
    fig.subplots_adjust(bottom=.15, left=.1)
    finish(fig, "fig3_confusion_matrix.png", "Figure 3: Confusion matrix on the unseen test session (row-normalised).")


# ---------------- Fig 4: weakest classes (precision / recall / F1) ----------------
def fig_weak(tp):
    rows = []
    for c in sorted(set(tp.y_true) | set(tp.y_pred)):
        tpn = ((tp.y_true == c) & (tp.y_pred == c)).sum()
        p = tpn / max((tp.y_pred == c).sum(), 1); r = tpn / max((tp.y_true == c).sum(), 1)
        rows.append((c, p, r, 2 * p * r / max(p + r, 1e-9)))
    d = pd.DataFrame(rows, columns=["cls", "p", "r", "f1"]).sort_values("f1").head(6)
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off")
    t = ax.table(cellText=[[r.cls, f"{r.p:.2f}", f"{r.r:.2f}", f"{r.f1:.2f}"] for r in d.itertuples()],
                 colLabels=["Class", "Precision", "Recall", "F1"], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(TABLE); t.scale(1, 2.1)
    fig.subplots_adjust(bottom=.18)
    finish(fig, "fig4_weakest_classes.png", "Table 1: The six hardest classes on the test session.")


# ---------------- Fig 5: live webcam trials ----------------
def fig_live(lt):
    lt = lt.assign(acc=lt.correct / lt.trials * 100)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.bar(lt.letter, lt.acc, color=[GREEN if a >= 80 else ORANGE for a in lt.acc])
    total = lt.correct.sum() / lt.trials.sum() * 100
    ax.axhline(total, color=GREY, lw=3, ls="--"); ax.text(len(lt) - .5, total + 2, f"Overall {total:.0f}%", ha="right", fontsize=TICK)
    ax.set_ylabel("Correct (%)"); ax.set_ylim(0, 112); ax.set_xlabel("Letter / gesture")
    ax.set_xticks(range(len(lt))); ax.set_xticklabels(lt.letter, rotation=90)
    fig.subplots_adjust(bottom=.27, left=.09)
    finish(fig, "fig5_live_trials.png", "Figure 4: Accuracy in repeated live webcam trials, per letter.")


# ---------------- Fig 6: NLP keystroke savings ----------------
def fig_nlp(nl):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.bar(nl.condition, nl.keystrokes, color=[GREY, GREEN], width=.55)
    base = nl.keystrokes.iloc[0]
    for i, k in enumerate(nl.keystrokes):
        ax.text(i, k + base * .02, f"{k:g}", ha="center", fontsize=AX)
    saved = (1 - nl.keystrokes.iloc[1] / base) * 100
    ax.set_ylabel("Letters signed"); ax.set_ylim(0, base * 1.15)
    ax.set_title(f"{saved:.0f}% fewer letters", fontsize=AX, color=GREEN)
    fig.subplots_adjust(bottom=.2, left=.14)
    finish(fig, "fig6_word_prediction.png", "Figure 5: Letters needed to type the same test sentences, with and without word prediction.")


def load(name):
    p = os.path.join(RES, name)
    return pd.read_csv(p) if need(p) else None


if __name__ == "__main__":
    fig_dataset()
    if DEMO:
        mc, tp, lt, nl = demo_frames()
    else:
        mc, tp, lt, nl = (load(n) for n in ("model_comparison.csv", "test_predictions.csv",
                                            "live_trials.csv", "nlp_eval.csv"))
    if mc is not None: fig_models(mc)
    if tp is not None: fig_confusion(tp); fig_weak(tp)
    if lt is not None: fig_live(lt)
    if nl is not None: fig_nlp(nl)