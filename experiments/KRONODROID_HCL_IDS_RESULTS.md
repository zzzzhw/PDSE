# HCL + IDS on KronoDroid

## Dataset and protocol

KronoDroid was selected because it is a timestamped Android dataset with both
benign and malicious applications and malware-family labels. The dataset was
published in *Computers & Security* (DOI
[`10.1016/j.cose.2021.102399`](https://doi.org/10.1016/j.cose.2021.102399))
and the pre-extracted CSV archives are distributed directly by the first
author in the [`aleguma/kronodroid`](https://github.com/aleguma/kronodroid)
repository. These CSV files require no access application. Raw APKs and system
call logs are access-controlled, but they are not needed for this experiment.

- Source commit: `c6ec342167bc449967a802824d068900ac8120c5`
- Real benign archive SHA-256: `21f6d507321856ee...70161bd5f782`
- Real malware archive SHA-256: `03aa36a9c3aa3430...817095cb8cd`
- Full real-device source: 78,137 rows (36,755 benign and 41,382 malware)
- Timestamp used: `HighestModDate`
- Initial training: 2011-01 through 2011-12
- Monthly test/adaptation: 2012-01 through 2012-12
- Initial training samples: 26,218 (23,085 benign, 3,133 malware, 94 families)
- Test samples: 7,909 (342 benign, 7,567 malware)
- Active-learning budget: 200 samples for each of the first 11 test months

Four SHA-256 values appeared once with each binary label. All eight ambiguous
rows were removed before temporal splitting. Model input excludes identity,
hash, timestamp, family, VirusTotal scanner/detection, submission-count, and
contacted-IP fields. Missing values are filled with zero, features are
transformed with `log1p`, and standardization is fitted only on 2011. Of 474
candidate intrinsic features, 226 are non-constant in the training window.

The HCL configuration follows the repository's large Drebin-feature setting:
SGD, learning rate 0.001, 200 initial epochs, Adam warm updates at 0.00001 for
50 epochs, batch size 1024, and the `hi-dist-xent`/local-pseudo-loss method.
Both methods use seed 1 and the same byte-identical initial checkpoint:
`da89fc8e450c7a1c98ca1a7c77473395f67cb81ac05fb9033c6aa55669f553c5`.

IDS uses the original static parameters: lambda 0.001, gamma 1.0, EMA decay
0.6, proxy learning rate 1.0, robust weight 1.0, batch size 192, warm-up 5,
and gradient clip 1.0.

## Results

| Metric | HCL | HCL + IDS | Difference |
| --- | ---: | ---: | ---: |
| Monthly mean F1 | 0.952742 | 0.953033 | +0.000292 |
| Monthly mean FNR | 0.087883 | 0.087358 | -0.000525 |
| Monthly mean FPR | 0.033575 | 0.033575 | 0.000000 |
| Monthly mean accuracy | 0.915258 | 0.915767 | +0.000508 |
| Minimum monthly F1 | 0.896500 | 0.896500 | 0.000000 |
| Final-month F1 | 0.984500 | 0.983100 | -0.001400 |
| AUT(F1), 12 test months | 10.492400 | 10.496600 | +0.004200 |
| Pooled F1 | 0.955461 | 0.955749 | +0.000288 |
| Pooled FNR | 0.084313 | 0.083785 | -0.000529 |
| Pooled FPR | 0.023392 | 0.023392 | 0.000000 |
| Pooled accuracy | 0.918321 | 0.918827 | +0.000506 |
| False negatives | 638 | 634 | -4 |
| False positives | 8 | 8 | 0 |

IDS improved monthly F1 in 5 months, reduced it in 2 months, and tied in 5
months. The largest gain was +0.0013 in 2012-04; the largest regression was
-0.0014 in 2012-12. A paired t-test over the 12 monthly F1 values gave
`p=0.2355`, and a Wilcoxon signed-rank test gave `p=0.3594`. The 95% confidence
interval for the mean paired F1 difference was `[-0.000220, 0.000803]`.

The pooled FPR is more reliable than the equal-weight monthly mean for this
stream. Some months have very few benign samples; for example, the 0.25 FPR in
2012-05 represents one false positive among four benign applications.

## Interpretation

The static IDS configuration produces a small directional improvement in mean
and pooled F1 and FNR, but the effect is not statistically significant and the
final month is worse. The evidence therefore does not support claiming a
stable KronoDroid performance improvement.

The main limiting factors observed in this run are:

1. HCL and HCL+IDS selected 98.23% of the same active-learning samples on
   average. IDS changes retraining slightly but barely changes the subsequent
   uncertainty ranking.
2. The 2012 stream is strongly malware-heavy. Binary balancing reduced the 200
   newly labeled samples to between 4 and 138 IDS center samples per month;
   early months provide a particularly weak benign exposure signal.
3. These IDS parameters were chosen for sparse APIGraph/Drebin features. The
   KronoDroid input is a standardized hybrid count representation, so the same
   perturbation scale is unlikely to be optimal.
4. HCL already reaches F1 values near 0.97-0.98 after adaptation, leaving a
   limited error set for IDS to correct.
5. The benchmark covers one seed and 12 test months. More seeds and a later
   temporal window are needed before drawing a general conclusion.

The 11 baseline monthly retrain phases took 397.6 seconds in total; IDS took
411.7 seconds, an overhead of about 3.6%. The full baseline process also spent
139.2 seconds training the shared initial checkpoint, so raw process runtimes
are not directly comparable.

## Correctness checks

- Six CPU/CUDA unit and integration tests passed.
- An end-to-end KronoDroid smoke run completed with 226 input features.
- The two initial checkpoints have identical SHA-256 hashes.
- All 11 expected IDS exposures and 495 active IDS epochs completed.
- No IDS epoch contained NaN or infinity in search loss, feature distance, or
  robust loss.

The reproducible scripts are `experiments/prepare_kronodroid.py` and
`experiments/run_kronodroid_comparison.py`. Compact raw comparisons are in
`experiments/results/kronodroid_real_2012/comparison.tsv` and
`experiments/results/kronodroid_real_2012/per_month.tsv`.
