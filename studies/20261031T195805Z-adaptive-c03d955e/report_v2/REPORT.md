# Adaptive replay study

Study: 20261031T195805Z-adaptive-c03d955e

Adaptive replay failed the fixed working criterion: 2 of 10 seeds met all targets, with 8 required. Fixed replay passed in 9 seeds.

The study trained 40 neural models in 419.2 seconds. Peak process memory was 630.4 MiB. Evidence verification passed.

The combined target was at least 80% final accuracy, at least 5 points less earlier accuracy lost than current-only training, and at most 5 points lower new-task acquisition accuracy. All three must hold in the same seed.

One controller was fixed before scoring. Fresh test seeds: 10. There was no parameter selection. Earlier studies and their negative results remain unchanged.

| Method | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost |
| --- | ---: | ---: | ---: |
| Current only | 39.3% ± 2.8 | 93.4% ± 0.6 | 64.9 pp ± 3.1 |
| Fixed 50% replay | 84.6% ± 1.4 | 89.1% ± 0.8 | 5.4 pp ± 1.4 |
| Adaptive replay | 79.0% ± 3.5 | 88.0% ± 1.0 | 10.8 pp ± 3.8 |
| Full history | 86.6% ± 1.2 | 85.2% ± 1.0 | -1.7 pp ± 1.0 |

Values show means and sample standard deviations across seeds. Earlier accuracy lost is signed; negative values mean improvement.

Relative to fixed replay, adaptive replay changed mean final accuracy by -5.57 points, acquisition by -1.06 points, and earlier accuracy lost by +5.41 points. Positive loss change means more forgetting. These paired differences are descriptive, not significance tests.

The separate linear reference reached 88.3% ± 0.8. It uses a different fitting procedure and a model suited to these rules.

Individual adaptive outcomes:

- Seed 907: final accuracy 84.310%; retention improvement 58.516 points; acquisition change -5.436 points; failed: acquisition tolerance.
- Seed 929: final accuracy 77.507%; retention improvement 48.652 points; acquisition change -4.834 points; failed: final accuracy target.
- Seed 953: final accuracy 81.510%; retention improvement 61.719 points; acquisition change -5.013 points; failed: acquisition tolerance.
- Seed 977: final accuracy 80.729%; retention improvement 53.691 points; acquisition change -3.809 points; passed.
- Seed 1009: final accuracy 80.599%; retention improvement 49.453 points; acquisition change -4.769 points; passed.
- Seed 1031: final accuracy 72.445%; retention improvement 49.570 points; acquisition change -7.682 points; failed: final accuracy target, acquisition tolerance.
- Seed 1051: final accuracy 79.997%; retention improvement 58.613 points; acquisition change -4.167 points; failed: final accuracy target.
- Seed 1087: final accuracy 80.339%; retention improvement 58.418 points; acquisition change -7.031 points; failed: acquisition tolerance.
- Seed 1109: final accuracy 74.691%; retention improvement 50.508 points; acquisition change -5.664 points; failed: final accuracy target, acquisition tolerance.
- Seed 1151: final accuracy 78.255%; retention improvement 52.285 points; acquisition change -5.745 points; failed: final accuracy target, acquisition tolerance.

Verdicts use unrounded scores. Individual results show extra precision so near-threshold failures remain visible.

Separate target counts explain the outcome. Acceptance still requires all three together.

| Method | Final target | Retention target | Acquisition tolerance |
| --- | ---: | ---: | ---: |
| Fixed 50% replay | 10 | 10 | 9 |
| Adaptive replay | 5 | 10 | 4 |

## Compute and controller behavior

Every neural model used 576 updates and 36,864 training-example presentations with forward and backward passes. Adaptive replay added 60 probe calls and 7,680 forward-only example evaluations per seed. Probe evaluations made no gradient updates.

| Method | Mean training seconds | Mean controller seconds | Mean benchmark evaluation seconds | Mean memory work seconds | Mean probe evidence capture seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current only | 9.048 | 0.000 | 0.720 | 0.000 | 0.000 |
| Fixed 50% replay | 10.085 | 0.000 | 0.601 | 0.176 | 0.000 |
| Adaptive replay | 10.531 | 0.216 | 0.671 | 0.208 | 0.297 |
| Full history | 7.735 | 0.000 | 0.567 | 0.000 | 0.000 |

The mean replay share after the first task was 50.2%. Across all seeds, the controller increased the share 144 times and decreased it 118 times. It reset to the initial share at each task.

The experiment matches gradient updates, not total compute. Extra probes are charged to the adaptive learner. Evidence capture supports auditing and is reported separately. Benchmark validation and test scores never control training. Wall times depend on system load; these results do not establish lower energy use or cheaper learning.

## Verification and limits

Verification reproduced 40 final models and 600 probe model states. It checked 1,474,560 training-example draws, 76,800 probe evaluations, and 17,280 batch allocations.

Probe IDs matched the frozen sampling rule and belonged to retained training memory. Recorded predictions and threshold decisions matched saved model states. Data, task order, model size, and initial weights matched within each seed.

Training-memory accuracy can improve without improved generalization. The test scores measure unseen examples. These remain supervised synthetic tasks with supplied identities and linear boundaries. The result does not establish language learning, general concepts, or replacement of large-model pre-training.

The model does not discover new architectures or choose its own goals. It changes one training allocation using a rule supplied by the experimenter. The required seed count is a working criterion, not a statistical significance test.

The parent directory retains the fixed plan, source, data, model states, training draws, controller trace, and per-method metrics. CSV files contain every scored seed and controller observation.
