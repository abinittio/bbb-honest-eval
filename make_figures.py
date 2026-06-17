"""Render the report figures as PNGs into report/figures/ from saved results."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(REPO, "results")
FIG = os.path.join(REPO, "report", "figures")
os.makedirs(FIG, exist_ok=True)
TEAL, GREY, LGREY = "#0D7377", "#777777", "#bbbbbb"


def fig_exp1():
    e1 = pd.read_csv(f"{RES}/exp1_split_study_summary.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(e1.index, e1["auc_mean"], yerr=e1["auc_std"], capsize=5,
           color=[TEAL, GREY, LGREY])
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.7, 1.0)
    ax.set_title("Split strategy (5 seeds)")
    fig.tight_layout(); fig.savefig(f"{FIG}/exp1_splits.png", dpi=160); plt.close(fig)


def fig_exp4():
    rel = pd.read_csv(f"{RES}/exp4_reliability.csv")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for m in ["uncalibrated", "temperature", "platt"]:
        d = rel[rel.method == m].dropna()
        ax.plot(d["confidence"], d["accuracy"], marker="o", label=m)
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    ax.set_title("Reliability (scaffold split)"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{FIG}/exp4_reliability.png", dpi=160); plt.close(fig)


def fig_exp5():
    e5 = pd.read_csv(f"{RES}/exp5_model_comparison_raw.csv")
    mean = e5.groupby(["model", "split"]).test_auc.mean().unstack("split")
    order = ["ECFP+RF", "StereoGNN", "ECFP+LogReg", "GCN", "GIN"]
    mean = mean.reindex(order)[["random", "scaffold", "cluster"]]
    ax = mean.plot.bar(figsize=(6, 3.2), rot=18, color=[TEAL, GREY, LGREY])
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.7, 0.95)
    ax.set_title("Cross-model robustness (5 seeds)")
    plt.tight_layout(); plt.savefig(f"{FIG}/exp5_models.png", dpi=160); plt.close()


fig_exp1(); fig_exp4(); fig_exp5()
print("wrote figures to", FIG)
print(os.listdir(FIG))
