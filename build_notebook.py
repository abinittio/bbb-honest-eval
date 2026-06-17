"""Generate the submission notebook bbb_eval_study.ipynb (valid nbformat-4 JSON).

The notebook is the analysis layer: it imports the src library, downloads the
data, loads the saved 5-seed results, and renders every table and figure, plus a
live quick reproduction proving the pipeline executes. Heavy training was done by
the src/exp*.py scripts; the notebook documents and visualises it reproducibly.
"""
import json, os

REPO = os.path.dirname(os.path.abspath(__file__))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    out = []
    for i, ln in enumerate(lines):
        out.append(ln + ("\n" if i < len(lines) - 1 else ""))
    return out


cells = [
    md("# Honest Evaluation of a Blood-Brain-Barrier GNN",
       "",
       "ECS7037P group project (individual submission). This notebook reproduces",
       "the figures and tables of the report. It imports the project library in",
       "`src/`, downloads BBBP and B3DB, loads the saved 5-seed results, and",
       "renders every experiment, ending with a live quick reproduction.",
       "",
       "**Experiments:** (1) split-strategy study, (2) deduplicated external",
       "validation, (3) ablation, (4) calibration, (5) cross-model robustness."),

    code("import os, sys, warnings",
         "warnings.filterwarnings('ignore')",
         "sys.path.insert(0, 'src')",
         "import numpy as np, pandas as pd",
         "import matplotlib.pyplot as plt",
         "from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')",
         "import study",
         "RES = 'results'",
         "print('library loaded')"),

    md("## Data",
       "",
       "BBBP (MoleculeNet, ~2,000 compounds) for training/evaluation; B3DB",
       "(~7,800 compounds) as an external set. Both download on first run."),

    code("import urllib.request",
         "os.makedirs('data', exist_ok=True)",
         "targets = {",
         "    'data/BBBP.csv': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv',",
         "    'data/B3DB_classification.tsv': 'https://raw.githubusercontent.com/theochem/B3DB/main/B3DB/B3DB_classification.tsv',",
         "}",
         "for path, url in targets.items():",
         "    if not os.path.exists(path):",
         "        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})",
         "        open(path, 'wb').write(urllib.request.urlopen(req, timeout=90).read())",
         "graphs, labels, smiles = study.load_bbbp()",
         "print(f'BBBP featurised: {len(graphs)} compounds '",
         "      f'({int(labels.sum())} BBB+, {int(len(labels)-labels.sum())} BBB-)')"),

    md("## Experiment 1: split-strategy study",
       "",
       "The same model is retrained from scratch under random, scaffold, and",
       "cluster splits (5 seeds). Random splitting overstates ROC-AUC; scaffold",
       "splitting is the honest, harder setting. Code: `src/exp1_split_study.py`."),

    code("e1 = pd.read_csv(f'{RES}/exp1_split_study_summary.csv', index_col=0)",
         "display(e1.round(4))",
         "gap = e1.loc['random','auc_mean'] - e1.loc['scaffold','auc_mean']",
         "print(f'random - scaffold AUC gap: {gap:+.4f}')",
         "fig, ax = plt.subplots(figsize=(5,3))",
         "ax.bar(e1.index, e1['auc_mean'], yerr=e1['auc_std'], capsize=5,",
         "       color=['#0D7377','#777','#bbb'])",
         "ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.7,1.0)",
         "ax.set_title('Exp 1: AUC by split strategy'); plt.tight_layout(); plt.show()"),

    md("## Experiment 2: external validation, naive vs deduplicated",
       "",
       "The deployed 5-fold ensemble on B3DB, before and after removing every",
       "compound whose InChIKey appears in BBBP training. The 0.96 reproduces and",
       "survives exact-overlap removal. Code: `src/exp2_external_validation.py`."),

    code("e2 = pd.read_csv(f'{RES}/exp2_external_validation.csv')",
         "display(e2[['set','n','auc','acc','sensitivity','specificity']].round(4))",
         "print(f\"AUC drop from removing overlap: \"",
         "      f\"{e2.loc[0,'auc'] - e2.loc[1,'auc']:+.4f}\")"),

    md("## Experiment 3: ablation",
       "",
       "(a) Zeroing the 6 stereo features; (b) varying encoder depth under the",
       "scaffold split. Stereo gives no measurable benefit on this",
       "non-stereoselective endpoint; 2 layers beat 4 (over-parameterisation).",
       "Code: `src/exp3_ablation.py`."),

    code("e3 = pd.read_csv(f'{RES}/exp3_ablation_raw.csv')",
         "a = e3[e3.block=='A']",
         "stereo = (a.groupby(['split','feats']).test_auc.mean().unstack('feats'))",
         "stereo['stereo_gain'] = stereo['full'] - stereo['no_stereo']",
         "print('(a) Stereo features:'); display(stereo.round(4))",
         "cap = pd.concat([a[(a.split=='scaffold')&(a.feats=='full')].assign(num_layers=4),",
         "                 e3[e3.block=='B']])",
         "capt = cap.groupby('num_layers').test_auc.agg(['mean','std']).round(4)",
         "print('(b) Encoder depth (scaffold):'); display(capt)"),

    md("## Experiment 4: calibration",
       "",
       "Expected calibration error (ECE) and reliability before scaling, after",
       "temperature scaling, and after Platt scaling. The model is over-confident",
       "(fitted T = 4.30). Code: `src/exp4_calibration.py`."),

    code("rel = pd.read_csv(f'{RES}/exp4_reliability.csv')",
         "def ece(df):",
         "    d = df.dropna(); N = d['count'].sum()",
         "    return float((d['count']/N * (d['confidence']-d['accuracy']).abs()).sum())",
         "tbl = rel.groupby('method').apply(ece).rename('ECE').reindex(",
         "    ['uncalibrated','temperature','platt'])",
         "display(tbl.round(4).to_frame())",
         "fig, ax = plt.subplots(figsize=(4,4))",
         "ax.plot([0,1],[0,1],'k--',lw=1,label='perfect')",
         "for m in ['uncalibrated','temperature','platt']:",
         "    d = rel[rel.method==m].dropna()",
         "    ax.plot(d['confidence'], d['accuracy'], marker='o', label=m)",
         "ax.set_xlabel('confidence'); ax.set_ylabel('accuracy')",
         "ax.set_title('Exp 4: reliability diagram'); ax.legend(); plt.tight_layout(); plt.show()"),

    md("## Experiment 5: cross-model robustness benchmark",
       "",
       "The identical protocol over a 5-model panel via the registry. Every model",
       "loses AUC from random to scaffold; ECFP+RandomForest beats every GNN on",
       "every split. Code: `src/exp5_model_comparison.py`, `src/models.py`."),

    code("e5 = pd.read_csv(f'{RES}/exp5_model_comparison_raw.csv')",
         "mean = e5.groupby(['model','split']).test_auc.mean().unstack('split')",
         "order = ['ECFP+RF','StereoGNN','ECFP+LogReg','GCN','GIN']",
         "splits = ['random','scaffold','cluster']",
         "mean = mean.reindex(order)[splits]",
         "display(mean.round(3))",
         "ax = mean.plot.bar(figsize=(7,3.5), rot=20, color=['#0D7377','#999','#ccc'])",
         "ax.set_ylabel('ROC-AUC'); ax.set_ylim(0.7,0.95)",
         "ax.set_title('Exp 5: model x split ROC-AUC'); plt.tight_layout(); plt.show()"),

    md("## Live reproduction (quick)",
       "",
       "Proof the pipeline executes end-to-end: retrain the model under random vs",
       "scaffold at 1 seed / 20 epochs (the report uses 5 seeds / 120 epochs).",
       "Expect random AUC >= scaffold AUC, the core effect, in about a minute."),

    code("for split in ['random','scaffold']:",
         "    tr, va, te = study.get_split(split, len(graphs), smiles, seed=0)",
         "    r = study.train_eval(graphs, labels, tr, va, te, seed=0, max_epochs=20)",
         "    print(f\"{split:9s} test AUC = {r['test_auc']:.4f}\")"),

    md("## Conclusion",
       "",
       "A model reported at 0.96 AUC ranks competently and its figure reproduces,",
       "but under leakage-controlled evaluation it loses ~0.057 to scaffold",
       "splitting, gains nothing from its stereo features on this task, is",
       "over-parameterised, is badly miscalibrated (T=4.30), and is beaten by a",
       "half-second fingerprint baseline. Evaluation design, not architecture,",
       "drove much of the headline number."),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = os.path.join(REPO, "bbb_eval_study.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, f"({len(cells)} cells)")
