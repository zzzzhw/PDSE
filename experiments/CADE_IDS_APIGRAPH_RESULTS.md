# CADE + IDS on APIGraph

## Setup

- Integration commits: `10f5e93` and `f7e0a29`
- Experiment driver commit: `ace5d66`
- Training period: 2012-01 through 2012-12
- Test period: 2013-01 through 2018-12 (72 months)
- Active-learning budget: 200 samples per month
- CADE configuration: `cae`, hidden dimensions 512-384-256-128,
  triplet-MSE loss, Adam, cosine schedule, initial learning rate 0.001,
  250 initial epochs, and 50 monthly warm-start epochs at learning rate 0.00005
- Classifier: MLP 100-100, 50 initial and monthly epochs
- Random seed: 1
- IDS parameters: lambda 0.001, gamma 1.0, EMA decay 0.6,
  proxy learning rate 1.0, robust weight 1.0, batch size 192,
  warm-up 5 epochs, and gradient clip 1.0
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- Runtime: CADE 18458.0 seconds including shared initial training; CADE+IDS
  18060.6 seconds from the copied initial checkpoints (run in parallel)

The comparison uses newly rerun CADE and CADE+IDS streams. The two methods
start from byte-identical 2012 CAE and MLP checkpoints, use the same seed,
sample budget, selector, and temporal data. Only IDS is changed.

## Integration

The CADE OOD selector remains unchanged. For each newly labeled monthly batch,
IDS constructs triplets consisting of a current sample, a similar historical
sample, and an opposite-binary-class historical sample. The loader emits all
anchors, all positives, and all negatives in three contiguous blocks, matching
the positional contract of CADE's triplet loss.

The proxy search maximizes CADE's triplet-MSE task loss while constraining the
normalized embedding displacement. Its normalized encoder-weight difference
is accumulated with an EMA. Monthly robust training evaluates the same CADE
loss at the perturbed encoder, updates the CAE, and then removes the synthetic
perturbation. The decoder participates in reconstruction and robust training,
but IDS perturbs only encoder weights.

## Results

All values below are calculated over the 72 test months.

| Metric | CADE | CADE + IDS | Difference |
| --- | ---: | ---: | ---: |
| Mean F1 | 0.87630 | 0.89246 | +0.01615 |
| Minimum F1 | 0.68090 | 0.70720 | +0.02630 |
| Maximum F1 | 0.95580 | 0.95690 | +0.00110 |
| F1 variance | 0.002281 | 0.001986 | -0.000295 |
| Final-month F1 | 0.92040 | 0.93700 | +0.01660 |
| AUT(F1) | 62.1558 | 63.3106 | +1.1548 |
| Mean FNR | 0.15341 | 0.13648 | -0.01692 |
| Minimum FNR | 0.02600 | 0.02600 | 0.00000 |
| Maximum FNR | 0.35150 | 0.30920 | -0.04230 |
| FNR variance | 0.004851 | 0.004495 | -0.000355 |
| Mean FPR | 0.00889 | 0.00744 | -0.00145 |
| Minimum FPR | 0.00240 | 0.00170 | -0.00070 |
| Maximum FPR | 0.08590 | 0.07740 | -0.00850 |
| FPR variance | 0.000104 | 0.000078 | -0.000027 |
| Mean accuracy | 0.97738 | 0.98025 | +0.00288 |
| Mean precision | 0.91488 | 0.92866 | +0.01378 |

The pooled confusion-matrix F1 improves from 0.87870 to 0.89339. Pooled FNR
drops from 0.15582 to 0.14050, and pooled FPR drops from 0.00818 to 0.00684.
This corresponds to 425 fewer false negatives and 350 fewer false positives
over the full stream.

IDS improves F1 in 52 months, reduces it in 17 months, and ties in 3 months.
The largest improvement is +0.1104 in 2018-04; the largest regression is
-0.0212 in 2014-10. On the 45 months where baseline F1 is below 0.9, the mean
improvement is +0.01972 and IDS wins 32 months.

| Year | CADE mean F1 | CADE + IDS mean F1 | Difference |
| --- | ---: | ---: | ---: |
| 2013 | 0.90127 | 0.89918 | -0.00208 |
| 2014 | 0.85563 | 0.85255 | -0.00308 |
| 2015 | 0.87680 | 0.90498 | +0.02818 |
| 2016 | 0.89123 | 0.91425 | +0.02302 |
| 2017 | 0.87148 | 0.89868 | +0.02719 |
| 2018 | 0.86141 | 0.88512 | +0.02371 |

The monthly paired t-test gives p=4.05e-7 with a 95% confidence interval of
[0.01039, 0.02192] for the mean F1 difference. The monthly Wilcoxon test gives
p=2.01e-7. Monthly observations are temporally correlated, so a six-year block
analysis is also reported: the paired t-test gives p=0.0427 and a year-block
bootstrap gives a 95% percentile interval of [0.00595, 0.02608]. The year-level
Wilcoxon test is not significant (p=0.1563) with only six blocks.

## Correctness Checks

- Nine unit and CUDA integration tests pass.
- Both result files contain exactly 72 aligned months.
- Initial CAE SHA-256: `6fbea86f3af74ef2ce3077b49cf21adf48e3096125ff2133b1c3ef13406df08b`
- Initial MLP SHA-256: `204bf411c162af3613ee0e51461ca02d94de1b6f80cd731fbbeedea71de8b73d`
- Source and destination hashes match for both initial checkpoints.
- The IDS log contains 71 exposure months and 3195 active IDS epochs
  (`71 * (50 - 5)`), with no numeric NaN or infinity values.
- Neither full experiment log contains a traceback or runtime error.

## Interpretation

Static IDS slightly hurts CADE in 2013 and 2014, then produces consistent
gains from 2015 through 2018. This supports the intended long-horizon drift
adaptation interpretation: the perturbation-based robust objective matters
after the accumulated stream has moved far enough from the initial training
distribution. It also shows that IDS is not structurally tied to HCL; with a
loss-compatible triplet batch contract, it transfers to CADE and yields a
larger mean F1 gain than the prior HCL integration.

The main remaining limitation is that this comparison uses one seed and the
static IDS parameters inherited from HCL. Multi-seed experiments are needed
for a stronger uncertainty estimate, and CADE-specific tuning may further
improve the first two years without sacrificing the later gains.
