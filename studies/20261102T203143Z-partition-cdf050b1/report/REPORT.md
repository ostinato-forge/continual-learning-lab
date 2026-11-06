# Task partition study

Study: 20261102T203143Z-partition-cdf050b1

Separate output layers failed the fixed working criterion. 5 of 10 seeds met all three targets; 8 were required.

Independent networks passed the fixed working criterion. 10 of 10 seeds met all three targets; 8 were required.

Fixed 50% replay failed the fixed working criterion. 7 of 10 seeds met all three targets; 8 were required.

The study trained 50 neural models in 595.9 seconds under resource limits. Peak process memory was 343.0 MiB. Evidence verification passed.

The targets were at least 80% final accuracy, at least 5 points less earlier accuracy lost than the shared current-only model, and at most 5 points lower acquisition accuracy. Acquisition is accuracy just after learning each task. All three conditions must hold for the same seed.

Both architectures were fixed before scoring on fresh test seeds. There was no selection or parameter tuning on these scores. Earlier failed criteria remain unchanged.

| Method | Final test accuracy | Acquisition accuracy | Earlier accuracy lost |
| --- | ---: | ---: | ---: |
| Shared, current only | 40.0% ± 2.4 | 93.9% ± 0.5 | 64.7 pp ± 3.0 |
| Fixed 50% replay | 85.6% ± 1.0 | 89.3% ± 0.8 | 4.4 pp ± 1.1 |
| Separate output layers | 79.8% ± 1.1 | 94.4% ± 0.5 | 17.5 pp ± 1.4 |
| Independent networks | 94.0% ± 0.4 | 94.0% ± 0.4 | 0.0 pp ± 0.0 |
| Full history | 87.4% ± 0.9 | 85.9% ± 0.6 | -1.8 pp ± 0.9 |

Values show means and sample standard deviations across seeds. Negative earlier loss means improvement.

The separate linear reference reached 88.6% ± 1.1. It has 216 fitted parameters and uses a different fitting procedure. Its training budget is not matched to the neural methods.

Relative to fixed replay, separate output layers changed final accuracy by -5.82 points, acquisition by +5.14 points, and earlier accuracy lost by +13.15 points. These are descriptive paired differences. Positive loss change means more forgetting.

Relative to fixed replay, independent networks changed final accuracy by +8.40 points, acquisition by +4.74 points, and earlier accuracy lost by -4.39 points. These are descriptive paired differences. Positive loss change means more forgetting.

| Method | Final target | Retention target | Acquisition tolerance | All targets |
| --- | ---: | ---: | ---: | ---: |
| Fixed 50% replay | 10 | 10 | 7 | 7 |
| Separate output layers | 5 | 10 | 10 | 5 |
| Independent networks | 10 | 10 | 10 | 10 |

Individual outcomes follow. Verdicts use unrounded scores; displayed precision is only for readability.

Separate output layers:

- Seed 1231: final 78.304%; retention improvement 44.824 points; acquisition change +0.944 points; failed: final_accuracy.
- Seed 1259: final 78.190%; retention improvement 48.242 points; acquisition change +0.391 points; failed: final_accuracy.
- Seed 1283: final 79.395%; retention improvement 50.820 points; acquisition change +0.472 points; failed: final_accuracy.
- Seed 1307: final 79.834%; retention improvement 40.957 points; acquisition change +0.374 points; failed: final_accuracy.
- Seed 1321: final 78.662%; retention improvement 46.680 points; acquisition change +0.472 points; failed: final_accuracy.
- Seed 1361: final 80.859%; retention improvement 48.594 points; acquisition change +1.009 points; passed.
- Seed 1399: final 80.941%; retention improvement 47.949 points; acquisition change +0.651 points; passed.
- Seed 1423: final 80.973%; retention improvement 48.730 points; acquisition change -0.146 points; passed.
- Seed 1451: final 80.111%; retention improvement 45.723 points; acquisition change +0.212 points; passed.
- Seed 1483: final 80.485%; retention improvement 49.043 points; acquisition change +0.505 points; passed.

Independent networks:

- Seed 1231: final 93.945%; retention improvement 64.512 points; acquisition change +0.179 points; passed.
- Seed 1259: final 93.473%; retention improvement 66.895 points; acquisition change +0.130 points; passed.
- Seed 1283: final 94.450%; retention improvement 69.043 points; acquisition change +0.342 points; passed.
- Seed 1307: final 94.173%; retention improvement 58.223 points; acquisition change +0.326 points; passed.
- Seed 1321: final 94.710%; retention improvement 66.426 points; acquisition change +0.065 points; passed.
- Seed 1361: final 93.522%; retention improvement 64.746 points; acquisition change +0.212 points; passed.
- Seed 1399: final 93.685%; retention improvement 64.121 points; acquisition change -0.081 points; passed.
- Seed 1423: final 93.538%; retention improvement 64.980 points; acquisition change -1.123 points; passed.
- Seed 1451: final 94.287%; retention improvement 61.914 points; acquisition change +0.895 points; passed.
- Seed 1483: final 94.173%; retention improvement 66.094 points; acquisition change -0.016 points; passed.

Fixed 50% replay:

- Seed 1231: final 86.117%; retention improvement 60.312 points; acquisition change -4.150 points; passed.
- Seed 1259: final 84.652%; retention improvement 61.602 points; acquisition change -4.281 points; passed.
- Seed 1283: final 84.342%; retention improvement 63.516 points; acquisition change -5.160 points; failed: acquisition.
- Seed 1307: final 86.930%; retention improvement 56.113 points; acquisition change -5.160 points; failed: acquisition.
- Seed 1321: final 86.751%; retention improvement 61.934 points; acquisition change -4.150 points; passed.
- Seed 1361: final 84.180%; retention improvement 60.527 points; acquisition change -5.615 points; failed: acquisition.
- Seed 1399: final 85.514%; retention improvement 59.648 points; acquisition change -4.525 points; passed.
- Seed 1423: final 86.214%; retention improvement 60.625 points; acquisition change -4.818 points; passed.
- Seed 1451: final 86.100%; retention improvement 58.633 points; acquisition change -4.557 points; passed.
- Seed 1483: final 85.124%; retention improvement 60.117 points; acquisition change -4.085 points; passed.

All task components count toward the total parameter budget, including untrained ones. Active parameters are those used for one example. Multiply-add counts cover dense layers in one forward pass and exclude routing, activation, optimizer, and backward work.

| Method | Total parameters | Active parameters | Forward multiply-adds per example | Retained examples |
| --- | ---: | ---: | ---: | ---: |
| Shared, current only | 70,660 | 70,660 | 70,144 | 0 |
| Fixed 50% replay | 70,660 | 70,660 | 70,144 | 1,024 |
| Separate output layers | 70,659 | 65,679 | 65,182 | 0 |
| Independent networks | 70,656 | 11,776 | 11,568 | 0 |
| Full history | 70,660 | 70,660 | 70,144 | 3,072 |

Every neural model used 576 updates and 36,864 training-example presentations. The new architectures used no replay or probes.

| Method | Training seconds | Evaluation seconds | Memory work seconds | Evidence seconds |
| --- | ---: | ---: | ---: | ---: |
| Shared, current only | 10.328 | 1.197 | 0.000 | 0.000 |
| Fixed 50% replay | 11.634 | 1.091 | 0.225 | 0.000 |
| Separate output layers | 10.482 | 1.480 | 0.000 | 0.264 |
| Independent networks | 7.534 | 0.540 | 0.000 | 0.573 |
| Full history | 10.294 | 1.138 | 0.000 | 0.000 |

Times are per-model wall-clock means under throttling. Pauses and other machine load affect them. They are not CPU or energy measurements and cannot be compared directly with earlier unthrottled studies. Evidence timing covers the added component audits and stage snapshots; baseline evidence overhead is not separately measured.

Verification reproduced 50 final checkpoints and 120 task-end states. It audited 1,843,200 training-example draws, checked 600 inactive components, and made 300 exact independent-task prediction comparisons.

The audit also compared 11,520 current-only batches across architectures and checked 11,520 baseline batch allocations. Saved linear weights reproduced all 10 references.

Data and task order matched within each seed. Shared baselines started with identical weights. Different architectures used the same seed and layer initialization procedure, but their weight tensors differ. Inactive task parameters stayed unchanged. Independent networks preserved other tasks' predictions exactly, as their structure requires.

The benchmark supplies task identity for routing. It uses six independent supervised tasks with simple linear rules. Fully separate networks cannot share learned features between tasks. Separate output layers still share changing hidden features. Neither approach in this study establishes transfer to new tasks or learned task recognition.

This result does not establish language learning, unlimited continual learning, lower total learning cost, or a replacement for large-model pre-training. The success threshold is a working criterion, not a statistical significance test. Further independent tasks would require another capacity decision.

The saved plan, source, data, checkpoints, batch IDs, component hashes, resource limits, and outcomes remain in the study directory. The launcher receipt separately records sampled resource use and any stop reason.
