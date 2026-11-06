# Replay allocation study

Study: 20261031T193006Z-allocation-91b72a62

Completed in 424.5 seconds. Trained 50 neural models. Evidence verification passed.

No tested replay share met the fixed validation target. Confirmation was skipped.

The rule requires at least 80% final accuracy, at least 5 percentage points less earlier accuracy lost than current-only training, and at most 5 points lower new-task acquisition accuracy. All three must hold in the same seed.

Selection required 8 of 10 validation seeds. Confirmation requires 4 of 5 fresh test seeds. No threshold was changed after scoring.

## Exploration

Test scores below describe the fixed comparison. Selection used validation scores only.

| Replay share | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost | Validation successes |
| --- | ---: | ---: | ---: | ---: |
| 0.0% | 38.6% ± 2.5 | 93.6% ± 0.3 | 66.0 pp ± 3.2 | reference |
| 12.5% | 61.4% ± 3.3 | 91.9% ± 0.7 | 36.6 pp ± 4.5 | 0 of 10 |
| 25.0% | 78.8% ± 1.1 | 90.5% ± 0.6 | 14.0 pp ± 1.6 | 0 of 10 |
| 50.0% | 85.5% ± 0.9 | 88.9% ± 0.5 | 4.1 pp ± 1.1 | 7 of 10 |

Values show means and sample standard deviations across seeds. Earlier accuracy lost is signed; negative values mean improvement.

Each validation condition counted separately:

| Replay share | Final accuracy target passed | Retention target passed | Acquisition tolerance passed |
| --- | ---: | ---: | ---: |
| 12.5% | 0 | 10 | 10 |
| 25.0% | 0 | 10 | 10 |
| 50.0% | 10 | 10 | 7 |

These diagnostic counts do not change the rule. All three conditions must pass together.

The exploration full-history reference reached 86.7% ± 1.1. The separate linear reference reached 88.2% ± 1.1.

## Cost and verification

Replay memory held up to 1,024 examples. Every neural model used 576 updates and 36,864 example presentations. Architecture, optimizer, and task distribution were fixed. The first task uses current examples only because no earlier memory exists.

Verification reproduced 50 final checkpoints, audited 1,843,200 training-example draws, and checked exact old/current counts in 23,040 batches. Data, task order, and starting weights matched across conditions within each seed.

Study time includes training, process startup, and validation selection. Confirmation readback and reporting occur afterward. Per-model training times are saved in CSV files. Equal update budgets do not establish equal energy use.

Memory size counts retained training examples. It excludes model weights, optimizer state, the current training block, and evaluation data. The benchmark saves full datasets for auditing; each learner's permitted access is checked from its batch trace.

## Limits

The selected share, if any, is only the smallest tested share meeting this target. The seed thresholds are experimental rules, not statistical significance tests. A different task family may give different results.

These are supervised synthetic tasks with supplied machine identities and linear rule boundaries. The separate linear reference uses a different procedure and a model suited to the rules. This experiment does not establish language ability, autonomous exploration, general concept learning, or replacement of language-model pre-training.

Earlier negative findings remain unchanged. This study changes the replay allocation on fresh seeds. It does not revise the earlier memory-size study's failed criterion.

## Files

- results.png and results.svg contain the charts.
- exploration.csv contains every exploration seed's validation and test metrics.
- confirmation.json contains individual outcomes and paired reference differences, or the reason confirmation was skipped.
- verification.json and validation_breakdown.json contain audit counts and diagnostic counts.
- The parent directory retains the frozen plan, source, data, selection, batch traces, and checkpoints.
