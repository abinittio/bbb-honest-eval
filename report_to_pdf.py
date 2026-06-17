"""Build a plain, phone-readable PDF of the report (not the IEEE-formatted one).

This is a review copy so the content can be checked on a phone. The submission
PDF is produced by compiling report/bbb_eval_report.tex on Overleaf.
"""
import os
from fpdf import FPDF

REPO = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(REPO, "report", "figures")
TEAL = (0, 0, 0)


class PDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 8)
        self.set_text_color(150)
        self.cell(0, 8, f"Review copy (not the IEEE submission PDF)  -  page {self.page_no()}",
                  align="C")


def h1(pdf, t):
    pdf.set_font("Times", "B", 15); pdf.set_text_color(*TEAL)
    pdf.multi_cell(pdf.epw, 8, t); pdf.set_text_color(0); pdf.ln(1)


def h2(pdf, t):
    pdf.ln(1); pdf.set_font("Times", "B", 12); pdf.set_text_color(*TEAL)
    pdf.multi_cell(pdf.epw, 7, t); pdf.set_text_color(0)


def body(pdf, t):
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(pdf.epw, 5.5, t); pdf.ln(1)


def table(pdf, rows, widths):
    pdf.set_font("Courier", "", 9)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            pdf.set_font("Courier", "B" if r == 0 else "", 9)
            pdf.cell(widths[c], 6, str(cell), border=1)
        pdf.ln()
    pdf.ln(2)


def image(pdf, name, w=120):
    p = os.path.join(FIG, name)
    if os.path.exists(p):
        pdf.image(p, w=w); pdf.ln(2)


pdf = PDF()
pdf.set_auto_page_break(True, margin=16)
pdf.add_page()

pdf.set_font("Times", "B", 16)
pdf.multi_cell(pdf.epw, 8, "How Much of a Blood-Brain-Barrier GNN's Performance "
                     "Survives Honest Evaluation?")
pdf.set_font("Times", "I", 10)
pdf.multi_cell(pdf.epw, 6, "Nabil Yasini-Ardekani  -  ECS7037P, QMUL  -  REVIEW COPY")
pdf.ln(2)

h2(pdf, "Abstract")
body(pdf, "We audit a stereochemistry-aware GNN for blood-brain-barrier (BBB) "
          "permeability that reports a 0.96 ROC-AUC external validation, and ask how "
          "much survives leakage-controlled evaluation. On BBBP (2,039 compounds) and "
          "B3DB (7,805): (i) random splitting overstates ROC-AUC by ~0.057 vs scaffold "
          "splitting; (ii) the 0.966 external number reproduces but 21.6% of the "
          "'external' set overlaps training, though removing it costs only 0.007 AUC; "
          "(iii) the stereo features give no measurable benefit on this "
          "non-stereoselective endpoint and a 2-layer encoder beats the 4-layer one "
          "(over-parameterised); (iv) the model is badly miscalibrated (T=4.30, ECE 0.11). "
          "A five-model benchmark shows ECFP+RandomForest beats every GNN on every split.")

h1(pdf, "1. Introduction")
body(pdf, "Predicting BBB permeability is core to CNS drug discovery. A metric is only "
          "as trustworthy as the protocol behind it. Random splits leak structural "
          "similarity (near-identical scaffolds on both sides); scaffold splitting "
          "controls this. 'External' validation leaks when the external set contains "
          "training compounds. We take one model reporting ~0.96 AUC and ask how much "
          "survives honest evaluation, and whether its architecture is justified. We do "
          "not claim to discover the leakage phenomenon (it is established); we contribute "
          "a controlled audit and a reusable, registry-driven harness.")

h1(pdf, "2. Methodology")
body(pdf, "Data: BBBP (2,039 cpds, 76% BBB+); B3DB as external set. Only 31% of BBBP has "
          "a defined stereocentre. Featurisation: 21 node features (15 atomic + 6 stereo), "
          "18 edge features. Model: GATv2 encoder (4 layers, hidden 128) + classifier; "
          "AdamW, BCE, early stopping on val AUC. Exp 2 uses the deployed pretrained "
          "5-fold ensemble; Exp 1/3/5 retrain from scratch to isolate each variable. "
          "Splits: random, scaffold (Bemis-Murcko), cluster (KMeans on ECFP4); every model "
          "sees the same split per seed. Code is a small library with a model registry, so "
          "adding a model to the benchmark is one line.")

h1(pdf, "3. Experiments and results")

h2(pdf, "Exp 1: split strategy (5 seeds)")
table(pdf, [["Split", "ROC-AUC", "PR-AUC"],
            ["Random", "0.897+/-.022", "0.951"],
            ["Scaffold", "0.840+/-.026", "0.938"],
            ["Cluster", "0.854+/-.064", "0.921"]], [40, 40, 40])
body(pdf, "Random-minus-scaffold gap = +0.057.")
image(pdf, "exp1_splits.png", w=95)

h2(pdf, "Exp 2: external validation (deployed ensemble)")
table(pdf, [["Set", "N", "AUC", "Sens", "Spec"],
            ["Naive", "7805", "0.966", "0.979", "0.671"],
            ["Dedup", "6119", "0.959", "0.974", "0.661"]], [34, 24, 24, 24, 24])
body(pdf, "21.6% of B3DB overlaps BBBP training, yet removing it costs only 0.007 AUC. "
          "The 0.96 reproduces and is NOT exact-duplicate memorisation. High sensitivity "
          "with low specificity (0.66) foreshadows the calibration problem.")

h2(pdf, "Exp 3: ablation (3 seeds)")
table(pdf, [["Stereo", "Random", "Scaffold"],
            ["Full", "0.899", "0.848"],
            ["No stereo", "0.908", "0.868"]], [40, 40, 40])
body(pdf, "Stereo features give no measurable benefit (within seed noise ~0.025). Expected: "
          "passive BBB diffusion is stereo-insensitive and ~2/3 of BBBP is achiral "
          "(carrier-mediated transport, e.g. L-DOPA via LAT1, can be stereoselective but is "
          "the minority route). Depth (scaffold): 4 layers 0.848, 2 layers 0.860, "
          "1 layer 0.837 -> over-parameterised.")

h2(pdf, "Exp 4: calibration (scaffold split)")
table(pdf, [["Method", "ECE", "Brier"],
            ["Uncalibrated", "0.106", "0.109"],
            ["Temperature", "0.080", "0.101"],
            ["Platt", "0.070", "0.100"]], [44, 30, 30])
body(pdf, "Fitted temperature T=4.30 (>1 = over-confident). Scaling halves the ECE.")
image(pdf, "exp4_reliability.png", w=80)

h2(pdf, "Exp 5: cross-model benchmark (5 seeds)")
table(pdf, [["Model", "Random", "Scaffold", "Cluster"],
            ["ECFP+RF", "0.926", "0.863", "0.890"],
            ["StereoGNN", "0.897", "0.834", "0.856"],
            ["ECFP+LogReg", "0.909", "0.821", "0.872"],
            ["GCN", "0.867", "0.799", "0.814"],
            ["GIN", "0.864", "0.793", "0.799"]], [42, 30, 30, 30])
body(pdf, "ECFP+RandomForest beats every GNN on every split, training in under a second. "
          "Your StereoGNN is the best of the neural models on scaffold.")
image(pdf, "exp5_models.png", w=125)

h1(pdf, "4. Analysis and conclusion")
body(pdf, "The 0.96 reproduces and is not a duplicate-memorisation artefact, but it is a "
          "random-split-favourable number: scaffold splitting costs ~0.057, and the "
          "original 'beats 8 SOTAs' comparison was not like-for-like. AUC hides a positive "
          "bias (specificity 0.66) explained by over-confidence (T=4.30). The stereo "
          "features earn nothing on BBB (expected; they likely help on stereo-sensitive "
          "endpoints like transporter activity), so stereo should be an explicit optional "
          "mode. The model is over-parameterised, and a half-second ECFP+RF baseline beats "
          "it under honest evaluation. Evaluation design, not architecture, drove much of "
          "the headline. Deliverable: an honest characterisation plus a reusable harness "
          "that makes the audit one line of code per added model.")

out = os.path.join(REPO, "report", "bbb_eval_report_REVIEW.pdf")
pdf.output(out)
print("wrote", out, f"({os.path.getsize(out)//1024} KB, {pdf.page_no()} pages)")
