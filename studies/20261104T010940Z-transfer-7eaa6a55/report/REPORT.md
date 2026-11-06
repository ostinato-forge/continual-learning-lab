# Shared feature transfer study

Study: 20261104T010940Z-transfer-7eaa6a55

Learned frozen features failed the fixed transfer criterion: 4 of 10 seeds met every target; 8 were required.

The comparison uses 128 labelled training examples per target for learned features and 512 for the fresh reference. Random features use 128. Source learning uses 2048 labelled examples per seed and is charged separately.

The decision requires at least 80 percent test accuracy on each new task, at most a two-point mean deficit to the fresh reference, at least a five-point advantage over random features, and unchanged source predictions. These numeric thresholds are the scored protocol; pipeline settings are shown in plan.json.

All sample sizes and comparisons were fixed before scoring. No result selected another seed, threshold, or sample size. Values below are means and sample standard deviations across seeds.

| Method | Target examples per task | Mean target test accuracy |
| --- | ---: | ---: |
| Learned frozen features | 32 | 83.61 ± 3.38% |
| Random frozen features | 32 | 53.82 ± 2.67% |
| Fresh model | 32 | 70.15 ± 3.23% |
| Fixed 50% replay | 32 | 76.63 ± 4.82% |
| Independent networks | 32 | 69.57 ± 3.29% |
| Linear reference | 32 | 65.40 ± 2.43% |
| Learned frozen features | 128 | 91.90 ± 0.99% |
| Random frozen features | 128 | 69.85 ± 4.09% |
| Fresh model | 128 | 87.68 ± 0.77% |
| Fixed 50% replay | 128 | 87.81 ± 1.93% |
| Independent networks | 128 | 87.72 ± 0.90% |
| Linear reference | 128 | 79.63 ± 2.91% |
| Learned frozen features | 512 | 93.92 ± 0.88% |
| Random frozen features | 512 | 78.10 ± 1.06% |
| Fresh model | 512 | 93.64 ± 0.71% |
| Fixed 50% replay | 512 | 92.18 ± 1.29% |
| Independent networks | 512 | 93.76 ± 0.56% |
| Linear reference | 512 | 88.37 ± 1.80% |

The linear reference fits separate raw-input classifiers for the two targets. It does not learn the four source tasks. Its measured time below includes both fitting and test prediction, so it differs from the neural training-only timing.

| Target examples per task | Fitted parameters across two targets | Mean fit and prediction CPU seconds |
| ---: | ---: | ---: |
| 32 | 72 | 0.001547 |
| 128 | 72 | 0.001582 |
| 512 | 72 | 0.001680 |

## Primary decisions by seed

| Seed | Learned features | Fresh reference | Random features | Failed conditions |
| --- | ---: | ---: | ---: | --- |
| 1601 | 91.6992% | 94.0430% | 71.4844% | fresh_accuracy_tolerance |
| 1613 | 94.2871% | 94.6289% | 68.4570% | none |
| 1627 | 91.3574% | 93.7012% | 78.2715% | fresh_accuracy_tolerance |
| 1657 | 91.2598% | 93.5059% | 72.8027% | fresh_accuracy_tolerance |
| 1663 | 90.8691% | 93.3105% | 66.0156% | fresh_accuracy_tolerance |
| 1693 | 92.2363% | 93.9453% | 69.0918% | none |
| 1721 | 90.9668% | 94.1406% | 68.3105% | fresh_accuracy_tolerance |
| 1733 | 92.2363% | 92.1875% | 68.5059% | none |
| 1747 | 92.2852% | 92.8711% | 63.4277% | none |
| 1759 | 91.7969% | 94.0430% | 72.1680% | fresh_accuracy_tolerance |

decisions.json retains the exact values and each target score. Rounding never changes a decision. The seed count is a working criterion, not a significance test.

## Source retention and complete learning cost

This table uses 128 target examples per task in all three complete-stream methods. It charges the source phase once to each method. Fresh and random-feature controls did not learn and retain the source tasks, so they are excluded from this full-stream comparison.

| Method | Source accuracy before targets | Source accuracy after targets | Earlier accuracy lost in points |
| --- | ---: | ---: | ---: |
| Learned frozen features | 92.61 ± 0.74% | 92.61 ± 0.74% | 0.00 ± 0.00 |
| Fixed 50% replay | 92.61 ± 0.74% | 92.16 ± 0.86% | 0.46 ± 1.24 |
| Independent networks | 94.06 ± 0.44% | 94.06 ± 0.44% | 0.00 ± 0.00 |

| Method | Source training CPU seconds | Target training CPU seconds | Total training dense MACs |
| --- | ---: | ---: | ---: |
| Learned frozen features | 2.617 | 0.539 | 5,509,742,592 |
| Fixed 50% replay | 2.617 | 1.764 | 7,057,833,984 |
| Independent networks | 1.572 | 0.797 | 1,239,957,504 |

MACs are dense-layer multiply-add operations in forward and backward matrix products. They exclude biases, activations, routing, loss, optimizer updates, and memory work. CPU timing includes the measured optimizer work but remains subject to machine conditions. Neither is an energy measurement.

The two source models are physically trained once per seed and copied into independent sample-size branches. Costs are attributed once per complete scenario, rather than treating copied source learning as free. Adding all scenario costs would double count shared execution. Overall study time is reported separately.

## Paired comparisons at the fixed sample size

Learned frozen features at 128 examples are compared below with references at 512. Differences are learned features minus the reference.

| Reference | Target accuracy difference in points | Complete training dense MAC difference |
| --- | ---: | ---: |
| Fresh model | -1.74 ± 1.07 | different source-task scope |
| Fixed 50% replay | -0.28 ± 1.65 | -1,548,091,392 |
| Independent networks | -1.86 ± 0.95 | +4,269,785,088 |

A lower arithmetic count is not a useful-cost win when the method delivers worse accuracy or retention. The primary criterion tests later labelled-data use. It does not prove lower total learning cost.

## Resources and evidence

The study completed 20 source models and 150 target branches in 775.2 seconds. Each branch contains two target fits. Peak process memory was 338.3 MiB.

Verification regenerated 10 datasets, reproduced 380 task-end checkpoints, and audited 2,334,720 training-example draws. It checked 1,690 protected components, 17,280 matching current-only batches, and 360 source prediction pairs. It reproduced 30 linear references from the allowed labels.

Verification reads saved model states and reproduces their predictions. It audits deterministic draw traces and parameter protection. It does not retrain every optimizer update.

| Architecture | Total parameters | Active parameters | Parameter bytes |
| --- | ---: | ---: | ---: |
| shared | 69,960 | 64,980 | 279,840 |
| independent | 70,050 | 11,675 | 280,200 |

results.csv records source and target cost, available and sampled labels, optimizer tensor bytes, and replay counts. The branch stage records also separate sampling, optimizer setup, memory work, evaluation, and checkpoint capture. These are measured sections, not all process work. They exclude model construction, branch copies, final prediction comparisons, and some audit and file-writing work. Overall study time includes that work. Learner-state counts exclude the experiment harness datasets and saved evidence; process memory includes them.

The launcher used one library thread, half of one CPU at low priority, a 2 GiB address-space limit, a 1 GiB resident-memory watchdog, and a 4 GiB available-memory reserve. Its own receipt records sampled CPU and memory and any stop reason.

## Interpretation limits

The tasks deliberately share four hidden directions. The target pairs are new combinations of those directions. Classifiers still receive supervised labels and supplied task identities. The feature extractor stops learning after four source tasks, so this tests bounded transfer, not continued feature learning over an unlimited stream.

All new methods use the same optimizer reset rule at task boundaries. The fresh reference resets its model for each target and does not retain old tasks. The linear reference has a different fitting procedure and must remain visible when judging whether the neural system is worthwhile.

No outcome establishes language learning, a novel continual-learning method, or a replacement for large-model pre-training. Earlier studies and their negative outcomes remain unchanged.
