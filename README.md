# Quantized, pruned autoencoder anomaly trigger (QKeras + hls4ml)

A small research prototype: how far can a tiny autoencoder jet-anomaly detector be compressed
(QKeras quantization-aware training, magnitude pruning) before it stops working, and does the
compressed model survive conversion with hls4ml? **No FPGA synthesis was run**: hls4ml is used
for conversion and C-simulation only, and resource numbers are analytic proxies.

**Short version: this is a mixed-to-negative result, reported as such.**

1. **The detector is weak.** Scoring jets by autoencoder reconstruction error on 16 jet-substructure
   features gives a pooled AUC of 0.687 ± 0.018 (largest model, FP32, 5 seeds), against a target of
   0.80. A trivial score, the mean of x² of the standardized input (what an all-zero reconstruction
   would give), reaches 0.695. The autoencoders beat it on W and Z jets and lose to it on top jets.
2. **Heavy compression is possible, within noise.** Several configurations with far fewer
   weights match the FP32 baseline's pooled AUC within seed noise, for example the 115-parameter
   `16-3-16` model with 4-bit weights and no pruning (about 65 non-zero parameters and 262 weight
   bits, against 387 parameters / 12,384 bits for the FP32 baseline): AUC 0.691 ± 0.040 over 5 seeds.
   Since the baseline itself only ties the trivial score, "matches the baseline" is a weak statement.
3. **Compression adds instability.** Quantized models vary much more from seed to seed (std 0.03 to 0.04)
   than FP32 models (0.002 to 0.018), and 75% pruning often degrades or completely collapses the
   reconstruction.
4. **hls4ml conversion agrees with Keras** after two fixes described below: AUCs that differ by at most 0.0003
   for all 27 configurations, and 20 of 27 within one output LSB for 99% of outputs.
5. **Not assessed:** latency (needs synthesis), LUT/DSP counts, and anything on real hardware.

Everything below states where each number comes from. Numbers marked *(seed 42)* come from single
saved models; all others are means over seeds 42 to 46.

## 1. Task and data

- Dataset: OpenML `hls4ml_lhc_jets_hlf` (id 42468; the hls4ml jet-substructure benchmark, Zenodo DOI
  10.5281/zenodo.3602260): 830,000 jets, 16 high-level features, 5 classes (g, q, W, Z, top).
- **Normal = QCD (g + q)**, 328,191 jets, split 70/15/15 into train / validation / test
  (229,733 / 49,228 / 49,230). **W, Z and top (about 167k each) are evaluation-only.**
- Preprocessing is fitted on QCD train only: clip at the 0.1 and 99.9 percentiles, log1p on
  `mass_mmdt` and `multiplicity`, standardize.
- Anomaly score: per-jet mean squared reconstruction error. Pooled AUC = QCD test vs W+Z+top pooled.
- The data split is fixed (seed 42). The 5 seeds (42 to 46) vary initialization and training only,
  so the spread reported here does not include data-split variance.

## 2. Models and compression

| name | widths | params |
|---|---|---|
| `16-8-4-3-4-8-16` | baseline, 3-unit latent | 387 |
| `16-4-4-4-16` | | 188 |
| `16-3-16` | | 115 |

- The latent layer is linear (no ReLU). With a ReLU latent, a 2-unit bottleneck lost units and
  the loss froze (see section 6). The two 2-unit-latent models trained early on are reported in
  `results/fp32_baseline_all.csv` and are near chance (pooled AUC 0.579 and 0.591).
- **Architectures were chosen after looking at anomaly-class AUC on the test set.** The validation
  set contains only QCD jets, so there was no clean selection set. Treat the architecture choice as
  tuned on test.
- **Quantization (QKeras, quantization-aware fine-tuning, 20 epochs from the FP32 weights):** weights and
  biases at 8, 6 or 4 bits (`quantized_bits(b, 1, alpha=1)`); activations and the latent fixed at
  8 bits with wide integer ranges (ReLU range up to 16, latent ±8); inputs 8 bits, outputs 10 bits.
  The sweep variable "bits" means **weight bits only**.
- **Pruning:** magnitude (unstructured) pruning of kernels with a polynomial schedule to 0, 50 and 75%
  sparsity, with the quantizers kept in the loop, 30 epochs, starting from the quantized model.
  Biases are not pruned. Non-zero counts are taken after quantization, so rounding to zero counts.
- **Resource proxy (mode C):** non-zero multipliers, total weight bits (non-zero parameters × bits)
  and BOPs (Baskin et al.-style formula, see `src/estimate_resources.py`). These are not LUT or DSP counts.

## 3. Results

### 3.1 Baselines

FP32 pooled AUC (5-seed mean ± std, from `results/sweep_runs.csv`): `16-8-4-3-4-8-16` 0.687 ± 0.018,
`16-3-16` 0.665 ± 0.002 (seed check). Per class for the largest model: W 0.683, Z 0.687, top 0.690.

Trivial detector, score = mean(x²): **W 0.614, Z 0.648, top 0.822, pooled 0.695.**
Reading it per class: the largest FP32 autoencoder is above the trivial score on W and Z and
well below it on top jets (about −0.13). `16-3-16` is below it on all three classes. The
Govorkova et al. paper (references) also notes that the best reconstruction does not
necessarily give the best anomaly detection, which is consistent with what is seen here.

### 3.2 Compression sweep (27 configs × 5 seeds)

Full table: `results/compression_table.csv` (also printed by `python -m src.aggregate`). Terms:
- **collapsed**: r² of the reconstruction below 0.2, or more than 50% of outputs exactly zero.
  The model outputs (almost) nothing and scores about the trivial AUC. 5 of 27 configs collapsed in at
  least one seed.
- **degraded**: not collapsed, but r² below 0.9× the FP32 model's r². 16 configs had at least one degraded
  seed; 6 are full quality.
- These thresholds were chosen **after** seeing results. The split of one flag into collapsed and
  degraded was proposed before the 5-seed run; the numeric thresholds were not pre-registered.
  The aggregation prints every config, so nothing is hidden by the definition.

Pareto front over non-collapsed configurations (weight bits vs pooled AUC, mean ± std over 5 seeds):

| config | weight bits | non-zero params | r² | pooled AUC | vs trivial | degraded seeds |
|---|---|---|---|---|---|---|
| `16-3-16`, 4b, 75% sparse | 88.8 | 22.2 | 0.57 | 0.639 ± 0.042 | −0.055 | 100% |
| `16-3-16`, 4b, 50% | 165.6 | 41.4 | 0.76 | 0.674 ± 0.016 | −0.021 | 100% |
| `16-3-16`, 4b, 0% | 261.6 | 65.4 | 0.82 | 0.691 ± 0.040 | −0.004 | 0% |
| `16-4-4-4-16`, 6b, 50% | 633.6 | 105.6 | 0.89 | 0.694 ± 0.033 | −0.001 | 0% |
| `16-4-4-4-16`, 8b, 50% | 857.6 | 107.2 | 0.90 | 0.694 ± 0.034 | 0.000 | 20% |

Two of the five (the third and fourth rows) have no degraded seed; those are the 2 full-quality points the
aggregation reports (target: at least 3). **Caveats that matter:**
- No non-collapsed config beats the trivial detector on pooled AUC (0 of 22). The best ones tie it.
- Best-of-27 selection on the test set overstates the winners. The 5-seed std is 0.03 to 0.04 for
  the quantized configs, so most differences between neighbouring configs are not resolved.
- "Drop vs FP32" in the table is measured against each architecture's own FP32 model; the smaller
  architectures have weaker FP32 baselines. Against the 387-parameter baseline (0.687) the
  headline configs are within about 0.01 of it on the mean.
- Per class, W and Z improve over the trivial score in many configs (up to about +0.07 for W) and top
  jets are always worse (−0.013 to −0.17).
- Weight bits for the FP32 baseline are 387 × 32 = 12,384. The two best configs use about 47 and 20 times fewer.

### 3.3 hls4ml conversion (C-simulation only)

Keras vs hls4ml 1.3.0 (Vitis backend, `io_parallel`, latency strategy), 27 saved *(seed 42)* models,
10,000 QCD jets plus 5,000 per anomaly class:
- AUC differences between Keras and hls4ml are at most 0.0003 in all 27 configs.
- 20 of 27 configs have at least 99% of outputs within one output LSB (1/32). The 7 that fall just short
  (98.5% to 99.0%) are 6 unpruned models and one 50%-pruned model; for these the max difference is at most 0.28 and the
  99th percentile is 0.0625. The cause was not isolated (rounding-tie differences are the likely one).
- Two fixes were needed (`src/convert.py`): (1) hls4ml's default accumulator `ap_fixed<16,6>` wraps around at
  ±32, which gave differences of up to 8.6 and AUC mismatches of up to 0.14; the accumulators are now
  `fixed<32,16>`. (2) hls4ml drops the linear quantizer layers (`q_latent`, `q_output`), so their saturation
  and rounding never reached the hardware model; the preceding dense layer's result type is set to the
  quantizer's type instead. Before the fixes, only 3 of 27 agreed.
- On recent macOS (Apple clang with libc++), the generated C++ does not compile without patching the installed
  hls4ml headers; see `scripts/patch_hls4ml_macos.py`.

### 3.4 Resources and latency

Mode C proxies *(seed 42)*, e.g. `16-8-4-3-4-8-16` 8-bit unpruned: 333 multipliers, 2,992 weight bits, 27,701
BOPs; `16-3-16` 4-bit 75%: 23 multipliers, 92 weight bits, 1,075 BOPs. See `results/resource_estimates_C.csv`.
**Latency was not assessed**: the proxy only gives layer depth (2, 4 or 6). A cycle count needs
HLS synthesis (mode A, not run: no Vitis/Vivado) or a predictor (mode B via rule4ml; that code is
untested and its API call is unverified).

## 4. Status against the original targets

| target | status |
|---|---|
| FP32 baseline AUC ≥ 0.80 | **Not met** (0.687 ± 0.018, largest model; single-seed best 0.707). Trivial score: 0.695 |
| Quantized AUC drop < 2% | Evaluated as a mean over 5 seeds with spread. 13 non-collapsed configs with ≥60% fewer params have a mean pooled drop below 0.02, 6 also with a 2-standard-error margin. Not resolvable against a weak, noisy baseline |
| ≥ 60% fewer non-zero params | Met by many configs (up to 98% fewer weight bits) |
| hls4ml conversion agrees with Keras | Met for all 27 configs by AUC; 20 of 27 within 1 LSB for 99% of outputs |
| Latency < 100 cycles | **Not assessed** |
| LUT / DSP numbers | **Not produced** (proxy only) |
| Pareto plot with ≥ 3 points | 5 non-collapsed points, 2 counting full-quality configs only |
| Tests pass, no Vivado needed | 17 tests. In a clean-clone run (fresh venv from `requirements.txt`) 16 passed and the hls4ml conversion test failed intermittently: it used unseeded random weights and a threshold at the noise level. The test was then seeded and changed to assert AUC agreement and no large differences; re-run `pytest` to confirm |

## 5. Open questions and known problems

- **Unexplained AUC loss.** The unpruned `16-8-4-3-4-8-16` quantized at 8, 6 or 4 bits lost 0.04 to 0.07
  pooled AUC in earlier single-config runs, with large seed-to-seed spread (per-seed drops from 0.01 to
  0.17 at 8 bits). Wider weight range, longer fine-tuning and an extra-training control did not explain it.
- **Reconstruction quality and detection are only moderately related.** Across the 22 non-collapsed configs the correlation between mean r² and mean pooled AUC is about 0.5 (computed from the aggregate table; 0.28 including collapsed configs), and configs with r² of at least 0.8 still span pooled AUC 0.63 to 0.69. In single-seed runs some of the highest AUCs came from models with lower r² (one 4-bit 50%-pruned model scored 0.78 at r² 0.75), which did not hold up over 5 seeds. Whether reconstruction error is the right score for compressed models was not tested.
- **75% sparsity** frequently collapses or degrades the model; uniform per-layer sparsity on very small networks
  is probably too aggressive, not tested.
- **Selection effects** (section 2) and fixed data split (section 1).
- **Post-hoc thresholds** for collapsed/degraded (section 3.2).
- **Seed 42 snapshots** (hls4ml agreement, resource proxies, range and collapse checks) are single models.

## 6. What changed along the way (v1 to v2)

The first sweep (kept in `results/v1/`, config `configs/default_v1.yaml`) used activation and latent ranges that were
too narrow, so 15 to 32% of latent values saturated in the larger models. Together with unpruned-model AUC losses and
hls4ml mismatches, this led to: decoupling activation bits (fixed at 8) from weight bits, wider ranges, wide hls4ml
accumulators and latent/output precision fixes, and collapse diagnostics. A ReLU latent in the 2-unit models caused a
frozen loss and was replaced by a linear latent. The first pruning results (75%-pruned models scoring exactly
the trivial AUC) were an artifact of constant-zero outputs; this is why r² and the trivial detector are tracked.

## 7. Reproduce

Tested on macOS (Apple silicon), Python 3.11.18, TensorFlow 2.13.1 (`tensorflow-macos`), QKeras 0.9.0,
tensorflow-model-optimization 0.8.1, hls4ml 1.3.0, scikit-learn 1.5.2, scipy 1.13.1, numpy 1.24.3,
`quantizers<2` (1.2.2). TensorFlow 2.13 needs Python 3.10 or 3.11. Newer scipy/scikit-learn wheels failed to load on the
test machine, and `quantizers` 2.x breaks the hls4ml import.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt          # or environment.yml with conda
python scripts/smoke_test.py             # every line should say "ok"
python scripts/patch_hls4ml_macos.py     # macOS only; edits installed hls4ml headers, run once

python -m src.prepare_dataset            # downloads from OpenML, writes data/processed/
python -m src.train                      # FP32 baselines (results/fp32_baseline.csv)
python -m src.quantize                   # QKeras models (models/qkeras, results/quantization_table.csv)
python -m src.prune                      # pruned models (models/pruned, results/pruning_table.csv)
python -m src.collapse_check             # r2 / zero-output / trivial-detector check (seed-42 models)
python -m src.convert --grid             # Keras vs hls4ml agreement (results/hls_agreement.csv)
python -m src.estimate_resources --mode C --grid
python -m src.latency_report             # reports that latency was not assessed in mode C
python -m src.sweep --seeds 5            # about 30 minutes; resumable
python -m src.aggregate                  # results/compression_table.csv, results/pareto.png
jupyter lab results.ipynb
pytest
```

Other tools: `src/seed_check.py` (seed spread and a no-quantization control), `src/range_check.py`
(how often values hit quantizer ranges), `src/trace_compare.py` (layer-by-layer Keras vs hls4ml).

Layout: `configs/default.yaml`, `src/` (pipeline), `tests/`, `scripts/`, `results/` (CSVs and plots; `results/v1/` is
the superseded first sweep), `models/` (gitignored), `data/` (gitignored).

## References

Checked against arXiv, the publisher pages or OpenML on 16 November 2026.

- J. Duarte et al., "Fast inference of deep neural networks in FPGAs for particle physics", JINST 13, P07027 (2018),
  arXiv:1804.06913. (hls4ml.)
- C. N. Coelho Jr. et al., "Automatic heterogeneous quantization of deep neural networks for low-latency inference on
  the edge for particle detectors", Nature Machine Intelligence 3, 675 (2021), arXiv:2006.10159. (QKeras / AutoQKeras.)
- E. Govorkova et al., "Autoencoders on field-programmable gate arrays for real-time, unsupervised new physics
  detection at 40 MHz at the Large Hadron Collider", Nature Machine Intelligence 4, 154 (2022), arXiv:2108.03986.
  (Closest prior work; it reports anomaly detection in about 80 ns on a Xilinx VU9P.)
- B. Hawks et al., "Ps and Qs: Quantization-aware pruning for efficient low latency neural network inference",
  Frontiers in Artificial Intelligence 4:676564 (2021), arXiv:2102.11289.
- Dataset: OpenML `hls4ml_lhc_jets_hlf` (id 42468), HLS4ML LHC Jet dataset, Zenodo DOI 10.5281/zenodo.3602260.
- BOPs (bit operations): Baskin et al., UNIQ. Not re-checked here; verify before citing.
