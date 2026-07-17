# HCL + IDS Hyperparameter Tuning

## Objective

The target was an absolute mean F1 improvement of 0.01 over the APIGraph HCL
baseline. Since the baseline mean F1 is 0.9102 over 2013-01 through 2018-12,
the target mean F1 was 0.9202.

## Protocol

All experiments used the same 2012 HCL checkpoint, active-learning budget of
200 samples per month, seed 1, 100 monthly retraining epochs, and the original
HCL training and selection settings.

The search used three stages:

1. Screen 23 alternative static parameter configurations on 2013-01 through
   2013-07.
2. Validate the two leading perturbation scales on 2013-01 through 2015-12.
3. Compare the long-window winner with the existing 72-month default IDS run.

The explored dimensions were perturbation scale, embedding constraint weight,
EMA decay, proxy learning rate, IDS batch size, warm-up epochs, and four
interactions between the leading perturbation scales and other parameters.

## Screening Results

The leading seven-month alternatives were:

| Configuration | Mean F1 | Delta vs HCL | Mean FNR | Worst monthly delta |
| --- | ---: | ---: | ---: | ---: |
| lambda=5e-4 | 0.93210 | +0.00253 | 0.09706 | -0.0127 |
| lambda=3e-4 | 0.93196 | +0.00239 | 0.09707 | -0.0059 |
| lambda=5e-4, proxy_lr=0.1 | 0.93189 | +0.00231 | 0.09714 | -0.0121 |
| proxy_lr=0.1 | 0.93011 | +0.00054 | 0.10137 | -0.0249 |
| batch_size=384 | 0.93007 | +0.00050 | 0.09994 | -0.0098 |

Larger gamma, larger EMA, larger perturbations, immediate IDS activation, and
more IDS optimization steps all caused material negative transfer in at least
one month. The complete ranking is stored in
`experiments/results/hcl_ids_tuning/to_2013-07/summary.tsv`.

## Long-Window Validation

| Configuration | Mean F1 | Delta vs HCL | Mean FNR | Wins | Worst delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| HCL baseline | 0.89852 | 0.00000 | - | - | - |
| Default IDS, lambda=1e-3 | **0.90689** | **+0.00836** | **0.14314** | 26/36 | -0.0367 |
| IDS, lambda=3e-4 | 0.90430 | +0.00578 | 0.14663 | 22/36 | -0.0433 |
| IDS, lambda=5e-4 | 0.90282 | +0.00430 | 0.14932 | 23/36 | -0.0556 |

The short-window winner did not generalize. Both reduced perturbation scales
lost important gains during the 2014 and 2015 drift periods. Therefore, they
were not promoted to a redundant 72-month run.

## Best Static Configuration

The best static configuration in the tested parameter space remains:

| Parameter | Value |
| --- | ---: |
| ids_lambda | 0.001 |
| ids_gamma | 1.0 |
| ids_ema_decay | 0.6 |
| ids_proxy_lr | 1.0 |
| ids_robust_weight | 1.0 |
| ids_batch_size | 192 |
| ids_warmup | 5 |
| ids_grad_clip | 1.0 |

Its full 72-month APIGraph mean F1 is 0.91512, an improvement of 0.00490 over
HCL. It does not reach the requested 0.9202 target and remains 0.00508 below
that threshold.

## Conclusion

The 0.01 target was not achieved by static hyperparameter tuning. The results
show that different drift periods require conflicting perturbation strengths:
smaller or less frequent perturbations reduce some regressions but also remove
the largest difficult-month gains. Further progress requires an adaptive IDS
gate or strength schedule based on the current labeled exposure set, rather
than a single global hyperparameter combination.
