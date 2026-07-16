# HCL + IDS on APIGraph

## Setup

- Integration commit: `af67990`
- W-DOE reference commit: `0987af3801eab744b98cdde9e53e85fe2fbe956b`
- Initial training period: 2012-01 through 2012-12
- Test period: 2013-01 through 2018-12 (72 months)
- Active-learning budget: 200 samples per month
- Initial checkpoint: the same 2012 HCL checkpoint used by the existing baseline
- Random seed: 1
- IDS parameters: lambda 0.001, gamma 1.0, EMA decay 0.6,
  proxy learning rate 1.0, robust weight 1.0, batch size 192,
  warm-up 5 epochs, gradient clip 1.0
- Hardware: NVIDIA GeForce RTX 4080 SUPER
- End-to-end runtime: 9980.3 seconds

The tracked reproduction command is in `experiments/scripts/base_hcl_ids.sh`.
The local raw result is `experiments/results/hcl_ids_full/apigraph.csv`.

## Results

| Metric | HCL baseline | HCL + IDS | Difference |
| --- | ---: | ---: | ---: |
| Mean F1 | 0.9102 | 0.9151 | +0.0049 |
| Minimum monthly F1 | 0.7820 | 0.8169 | +0.0349 |
| Monthly F1 standard deviation | 0.0360 | 0.0331 | -0.0030 |
| Mean FNR | 0.1268 | 0.1217 | -0.0051 |
| Mean FPR | 0.00444 | 0.00404 | -0.00040 |
| Mean accuracy | 0.98385 | 0.98473 | +0.00088 |
| Final-month F1 | 0.9745 | 0.9770 | +0.0025 |
| Final AUT(F1) | 65.5454 | 65.8976 | +0.3522 |

HCL + IDS improved F1 in 49 months, reduced it in 22 months, and tied in
1 month. For the 22 difficult months where baseline F1 was below 0.9, the
mean F1 improvement was 0.0157 and IDS won in 17 months.

The largest improvement was 0.0803 in 2014-10. The largest regression was
-0.0446 in 2018-09, so the method is not uniformly better month by month.

| Year | HCL mean F1 | HCL + IDS mean F1 | Difference | IDS wins |
| --- | ---: | ---: | ---: | ---: |
| 2013 | 0.9188 | 0.9244 | +0.0056 | 8/12 |
| 2014 | 0.8817 | 0.8860 | +0.0043 | 8/12 |
| 2015 | 0.8951 | 0.9102 | +0.0152 | 10/12 |
| 2016 | 0.9300 | 0.9312 | +0.0012 | 8/12 |
| 2017 | 0.9162 | 0.9191 | +0.0030 | 7/12 |
| 2018 | 0.9196 | 0.9198 | +0.0002 | 8/12 |

A paired t-test over monthly F1 values gave p=0.0271 and a Wilcoxon signed
rank test gave p=0.0090. The 95% confidence interval for the mean paired F1
difference was [0.00057, 0.00923].

## Correctness Checks

- CPU unit tests: 3 passed, 1 CUDA-only test skipped as expected.
- CUDA unit and integration tests: 4 passed.
- Python compilation checks passed.
- 71 monthly IDS exposures completed.
- 6745 IDS epoch records contained no NaN or infinity values.

The tests cover exposure triplet roles, binary balancing, proxy perturbation
generation, exact perturbation restoration, finite gradients and losses, and
the CUDA monthly-retraining path.

## Interpretation

This configuration improves APIGraph mean F1 and reduces both average FNR and
FPR. Its strongest effect is on difficult drift months, but it also causes
material regressions in a minority of months. A future iteration should make
the robust IDS weight depend on the composition or validation behavior of the
current active-learning batch instead of always using weight 1.0.

The comparison uses the repository's existing full HCL result. That baseline
was not rerun in this experiment with the new seed flag. Three existing HCL
runs are nevertheless tightly grouped (mean F1 from 0.91008 to 0.91021), and
the selected baseline uses the same initial checkpoint as HCL + IDS.
