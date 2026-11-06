# Memory capacity experiment

Study: 20261031T184727Z-fc9c477e

Completed in 252.9 seconds. Trained 80 neural models. Evidence verification passed.

## Selection and confirmation

No tested memory met the fixed validation criterion. Confirmation was therefore skipped.

Selection required at least 80 percent final accuracy, at least 5 percentage points less earlier accuracy lost than current-only training, and at most 5 points lower new-task acquisition accuracy, in 8 of 10 exploration seeds. Selection used validation scores only.

## Exploration results

These test scores describe the fixed capacity sweep. They were not inputs to memory selection.

| Memory examples | Final test accuracy | New-task acquisition accuracy | Earlier accuracy lost | Validation seeds meeting all targets |
| --- | ---: | ---: | ---: | ---: |
| 0 | 39.7% ± 2.5 | 93.8% ± 0.3 | 65.0 pp ± 3.2 | reference |
| 16 | 48.4% ± 2.4 | 91.5% ± 0.5 | 51.8 pp ± 3.0 | 0 |
| 64 | 61.7% ± 2.0 | 90.2% ± 0.5 | 34.3 pp ± 2.2 | 0 |
| 128 | 70.0% ± 1.6 | 89.9% ± 0.6 | 23.8 pp ± 2.1 | 0 |
| 256 | 78.2% ± 2.1 | 89.3% ± 0.6 | 13.4 pp ± 2.4 | 3 |
| 512 | 82.1% ± 1.3 | 89.1% ± 0.5 | 8.4 pp ± 1.5 | 6 |
| 1024 | 85.5% ± 0.9 | 89.3% ± 0.4 | 4.5 pp ± 1.3 | 7 |

Values show means and sample standard deviations across seeds. Earlier accuracy lost is a signed difference; negative values mean improvement.

The original 256-example comparison met its retention and acquisition criterion in 7 of 10 fresh seeds. The predeclared required count was 8. Replication criterion failed. This criterion does not include the new final-accuracy target.

The full-history reference reached 86.2% ± 0.8. The separate linear reference reached 88.7% ± 0.8.

## Cost, verification, and limits

Every neural model used 576 updates and 36,864 example presentations. Architecture, task distribution, optimizer, and replay fraction were held fixed.

Verification reloaded 80 checkpoints and audited 2,949,120 training-example draws. Exploration datasets, task orders, and starting weights were identical across capacities within each seed.

The complete study time includes condition runs, process startup, and validation selection. The final confirmation readback and report generation occur afterward. Per-model training times are in the CSV files. Similar timings do not establish identical energy use.

Memory counts describe retained example access. They do not include neural weights, optimizer state, the currently available training block, or the benchmark's evaluation data.

The selected size, if any, is only the smallest tested size meeting this particular target. It is not a universal minimum. A failure to meet a threshold is retained as a negative result.

These remain supervised synthetic tasks with explicit machine identities and linear rule boundaries. The simple linear reference uses a different learning procedure and 216 fitted parameters. The experiment does not establish language ability, autonomous exploration, general concept learning, or the elimination of large-model pre-training.

## Files

- results.png and results.svg contain the charts.
- exploration.csv contains both validation and test scores for every capacity and seed.
- confirmation.json contains the fresh-seed outcome, including failures or the reason it was skipped.
- verification.json records the evidence checks.
- The parent directory retains the frozen plan, source, input configurations, selection, and all condition runs.
