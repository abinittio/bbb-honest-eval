# Honest Evaluation of a Blood-Brain-Barrier GNN

A reproducible, leakage-controlled audit of a stereochemistry-aware graph neural
network (GNN) for blood-brain-barrier (BBB) permeability. The question it answers:
**how much of a model's reported performance survives honest evaluation?**

This is a methodology and reproducibility study. It takes a real model that reports
~0.96 ROC-AUC and stress-tests it the hard way, reporting what holds up and what
does not.

## Key findings

On the MoleculeNet BBBP set (2,039 compounds) and the B3DB external set (7,805):

- **Random splitting overstates ROC-AUC by ~0.057** versus scaffold splitting.
- The 0.96 external figure **reproduces** but ~22% of the "external" set overlaps
  training; removing it costs only 0.007, so the figure is real but
  evaluation-favourable, not a duplicate-memorisation artefact.
- The model is **functionally stereo-blind**: it returns identical predictions for
  enantiomers, because its features encode the presence of a stereocentre, not its
  R/S configuration.
- It is **over-parameterised** (a 2-layer encoder beats the 4-layer one) and
  **over-confident** (expected calibration error ~0.11; fitted temperature 4.3).
- A simple **ECFP fingerprint + random forest baseline outperforms it** on every
  split.

The headline: evaluation design, not architecture, drove much of the reported
number. The deliverable is an honest characterisation plus a reusable,
registry-driven harness for auditing other models the same way.

## Reproduce

```bash
pip install -r requirements.txt
python src/exp1_split_study.py          # split-strategy study
python src/exp2_external_validation.py  # naive vs deduplicated external test
python src/exp3_ablation.py             # stereo features + encoder depth
python src/exp4_calibration.py          # ECE / Brier, temperature + Platt
python src/exp5_model_comparison.py     # cross-model robustness benchmark
```
Datasets download automatically on first run. Full write-up in
`report/bbb_eval_report.tex`; analysis notebook in `bbb_eval_study.ipynb`.

## Cite

If you use this work, please cite it (see `CITATION.cff`).

## License

MIT (see `LICENSE`). Attribution is requested via `CITATION.cff`.
