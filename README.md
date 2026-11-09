# Continual learning lab

> **Status:** A research lab, not a product. Five studies have run under preregistered protocols; each study folder holds its plan, decision rule, report, and chart, with the raw run data left out of this repository. Needs Python 3.12 or later and PyTorch on CPU. Maintained.

Run a small neural model on a sequence of new rules. Compare learning from current
examples, replaying a small memory, retaining all earlier examples, separating
task networks, and reusing learned features.

The first experiment asks whether memory replay reduces forgetting at a fixed
training budget. It starts from random weights and uses generated numerical
inputs with labels supplied by a simulator.

Read [PROTOCOL.md](PROTOCOL.md) for the rules fixed before the first scored run.
The main configuration is [configs/pilot.json](configs/pilot.json).

## Resource limits

Run tests, experiments, and report verification through the limited launcher:

    python3 limited_run.py .venv/bin/python -m unittest discover -s tests -p test_transfer.py -v
    python3 limited_run.py .venv/bin/python transfer_study.py --plan PATH_TO_NEW_PLAN
    python3 limited_run.py .venv/bin/python analyze_transfer.py STUDY_DIRECTORY

The three launcher tests run separately because they create limited child
processes themselves. Do not include them in a test suite inside the launcher.

    .venv/bin/python -m unittest discover -s tests -p test_resource_limits.py -v

The default uses one CPU at low priority and allows execution for half of each
200-millisecond interval. Numerical libraries use one thread. Each process has
a hard 2 GiB address-space limit. A watchdog stops the job if sampled resident
memory exceeds 1 GiB or available system memory falls below 4 GiB. Resident
checks can briefly overshoot. The launcher refuses a second concurrent lab job.

Logs, limits, resource samples, and the exit reason are saved in checks/limited.
The launcher prints the directory at startup. Stop its process with Ctrl+C to
interrupt the child and retain an outcome receipt. A process killed by the
memory limit can leave incomplete experiment evidence. Keep that evidence.

The partition and transfer studies use one PyTorch thread and refuse to start
without the launcher. Older study commands below should use the same launcher
prefix. The older frozen study source remains unchanged. A new scored study
needs a copied plan with unused seeds and a decision rule fixed before scoring.

## Shared feature transfer study

Earlier learning helped with new related tasks, but failed the fixed transfer
criterion. Four of ten seeds met all conditions; eight were required.

The model first learned four tasks that shared hidden rules. It then kept its
hidden layers fixed and learned output layers for two new combinations of
those rules. Each method received raw inputs, labels, and task identities.

| Method | Labelled examples per new task | Mean test accuracy on new tasks |
| --- | ---: | ---: |
| Learned frozen features | 128 | 91.90% |
| Random frozen features | 128 | 69.85% |
| Fresh model | 128 | 87.68% |
| Fresh model | 512 | 93.64% |
| Independent networks | 512 | 93.76% |
| Linear reference | 512 | 88.37% |

The learned-feature method kept all earlier predictions unchanged. Every seed
also passed the 80 percent per-task accuracy floor and the required advantage
over random features. Six seeds failed to stay within two accuracy points of
the fresh model with 512 labels. Their deficits were 2.25 to 3.17 points.
Although the mean deficit was 1.74 points, only four seeds passed the complete
rule. All results at 32, 128, and 512 labels remain in the report.

Initial learning used 2,048 labelled examples and is included in each complete
scenario's cost. With 128 labels per new task, measured source-plus-target
training took 3.156 CPU seconds for learned features and 2.369 for independent
networks. The learned-feature model used about 4.44 times the estimated dense
training arithmetic. That arithmetic ratio is not a runtime ratio. The study
does not establish lower total learning cost.

All 58 scientific tests passed. Verification reproduced 380 checkpoints,
audited 2,334,720 training draws, and confirmed data separation, matching
current-only batches, protected parameters, and saved decisions. Training
peaked at 338.3 MiB and completed under the unchanged resource limits. The
report states which costs are measured and what verification does not test.

Read the [transfer report](studies/20261104T010940Z-transfer-7eaa6a55/report/REPORT.md),
view the [chart](studies/20261104T010940Z-transfer-7eaa6a55/report/results.png),
or inspect the [fixed protocol](TRANSFER_STUDY.md).

The proposed next test uses one smaller shared feature model and stores frozen
features for reuse during training. It must meet a fixed total cost budget,
including initial learning, while preserving accuracy and retention. This
follow-up is not yet run. The existing result concerns one synthetic task
family and does not establish a replacement for large-model pre-training.

## Task partition study

Independent small networks reached 94.0 percent final test accuracy with no
measured forgetting. They met the fixed success rule in all ten fresh seeds.
All task networks together used 70,656 parameters, compared with 70,660 for
the shared reference. They stored no old training examples.

| Method | Final test accuracy | Acquisition accuracy | Earlier accuracy lost | Seeds meeting all targets |
| --- | ---: | ---: | ---: | ---: |
| Shared, current only | 40.0% | 93.9% | 64.7 points | reference |
| Fixed 50% replay | 85.6% | 89.3% | 4.4 points | 7 of 10 |
| Separate output layers | 79.8% | 94.4% | 17.5 points | 5 of 10 |
| Independent networks | 94.0% | 94.0% | 0.0 points | 10 of 10 |
| Full history | 87.4% | 85.9% | -1.8 points | reference |

Eight successes were required. Separate output layers missed the final-accuracy
target in five seeds. Fixed replay exceeded the acquisition-loss allowance in
three seeds. All earlier results remain unchanged. The separate linear
reference reached 88.6 percent with a different fitting procedure.

Each model received 576 updates and 36,864 training-example presentations.
All methods that used only current examples received identical batches.
Verification reproduced 50 final checkpoints and 120 task-end states and
audited 1,843,200 training-example draws. All 46 scientific checks and three
separate resource-limit checks passed before scoring.

The study completed in 595.9 seconds. The training process peaked at 343 MiB.
Including startup and exit, the launcher recorded 611.3 seconds elapsed and
154.6 sampled CPU seconds, about 25 percent of one CPU core on average.
It used low priority, one thread, and a half-core throttle. Wall times are not
directly comparable with earlier unthrottled studies.

This result fits the benchmark's structure. Task identity is supplied, and the
tasks are independent. Separate networks preserve old predictions because
learning one task never changes another task's weights. They do not share
learned knowledge. Adding more tasks would require another capacity decision.

Read the [study report](studies/20261102T203143Z-partition-cdf050b1/report/REPORT.md),
view the [chart](studies/20261102T203143Z-partition-cdf050b1/report/results.png),
or inspect the [fixed plan](PARTITION_STUDY.md).

The transfer study above tests reuse on related tasks. More variants of the
independent-rule benchmark are unlikely to answer the broader question about
learning without large pre-training runs.

## Adaptive replay study


Adaptive replay performed worse than fixed 50 percent replay on ten fresh test
seeds. The controller changed the share of old examples using accuracy on
retained training memory. Its rule was fixed before scoring.

| Method | Mean final test accuracy | Mean acquisition accuracy | Mean earlier accuracy lost | Seeds meeting all targets |
| --- | ---: | ---: | ---: | ---: |
| Current only | 39.3% | 93.4% | 64.9 points | reference |
| Fixed 50% replay | 84.6% | 89.1% | 5.4 points | 9 of 10 |
| Adaptive replay | 79.0% | 88.0% | 10.8 points | 2 of 10 |
| Full history | 86.6% | 85.2% | -1.7 points | reference |

Eight successes were required. Adaptive replay failed that criterion. It used
50.2 percent replay on average after the first task and added 7,680 forward-only
probe evaluations per model. The gradient update budget stayed fixed. This
controller gave worse mean accuracy and retention with extra monitoring work.

The fixed method passed in nine seeds in this direct test. Earlier results used
different seeds, and the two earlier follow-ups selected using validation data.
Their negative outcomes remain unchanged.

The study trained 40 models in 419.2 seconds, with 630.4 MiB peak process memory.
All 35 software checks passed before scoring. Verification reproduced 40 final
models and 600 probe model states, audited 1,474,560 training-example draws,
and checked the controller's access and decisions.

Read the [adaptive report](studies/20261031T195805Z-adaptive-c03d955e/report_v2/REPORT.md),
view the [charts](studies/20261031T195805Z-adaptive-c03d955e/report_v2/results.png),
or inspect the [fixed controller plan](ADAPTIVE_REPLAY_STUDY.md).
The first report remains preserved. Revision two adds precise individual scores
and names failed targets so rounding cannot hide near-threshold failures.
No scores, thresholds, or decisions changed.

To run another study, copy configs/adaptive_replay.json and choose unused seeds.
The adaptive runner rejects seeds already present in scored run configurations.

    .venv/bin/python adaptive_study.py --plan PATH_TO_NEW_PLAN
    .venv/bin/python analyze_adaptive.py STUDY_DIRECTORY

## Replay allocation study

The next study fixed memory at 1,024 examples and compared the share of each
later batch devoted to old examples. It trained 50 neural models on ten fresh
seeds in 424.5 seconds. Model size and total training budget stayed fixed.

| Replay share | Mean final test accuracy | Mean new-task acquisition accuracy |
| --- | ---: | ---: |
| 0% | 38.6% | 93.6% |
| 12.5% | 61.4% | 91.9% |
| 25% | 78.8% | 90.5% |
| 50% | 85.5% | 88.9% |

Less replay improved new learning but reduced retention. Both lower shares
missed the 80 percent final validation target in every seed. Half-batch replay
passed that target and the retention target in every seed, but exceeded the
allowed five-point new-learning penalty in three seeds.

No share met all targets in the required eight of ten validation seeds.
Confirmation was skipped as planned. The earlier negative memory-size finding
remains unchanged. These results do not show a reduction in compute cost.

Read the [allocation report](studies/20261031T193006Z-allocation-91b72a62/report/REPORT.md),
view the [charts](studies/20261031T193006Z-allocation-91b72a62/report/results.png),
or inspect the [fixed allocation plan](REPLAY_ALLOCATION_STUDY.md).

    .venv/bin/python replay_allocation.py --plan configs/replay_allocation.json
    .venv/bin/python analyze_replay_allocation.py report STUDY_DIRECTORY

All 27 software checks passed. Verification reproduced all 50 checkpoints,
audited 1,843,200 training-example presentations, and checked the exact replay
share in 23,040 batches. The separate tiny pipeline check exercised selection,
confirmation, and the paired reference. Its scores are not research evidence.

## Memory size study

The follow-up trained 80 neural models with ten fresh exploration seeds.
It completed in 252.9 seconds and tested replay memories of 0, 16, 64, 128, 256,
512, and 1,024 examples at the same training budget.

More memory improved retention. Final test accuracy increased from 39.7 percent
without memory to 85.5 percent with 1,024 stored examples. The full-history
reference reached 86.2 percent.

No tested size met all fixed validation targets in eight of ten seeds.
The largest two sizes passed the final-accuracy and retention targets in every
seed, but exceeded the allowed new-learning penalty in four and three seeds.
Confirmation on additional seeds was skipped as specified in the plan.

The original 256-example comparison also missed its repetition target:
seven of ten successes, with eight required.

Read the [memory study report](studies/20261031T184727Z-fc9c477e/report_v2/REPORT.md),
view the [memory study charts](studies/20261031T184727Z-fc9c477e/report_v2/results.png),
or inspect the [fixed study plan](MEMORY_STUDY.md).
The original report remains in the same study's report directory. Revision two
adds the validation-condition breakdown and corrects the chart caption.
The experimental results and selection are unchanged.

Run the study with:

    .venv/bin/python memory_sweep.py --plan configs/memory_sweep.json
    .venv/bin/python analyze_memory_sweep.py report STUDY_DIRECTORY

All 18 software checks passed. Verification reproduced the 80 saved neural
models and audited 2,949,120 training-example presentations.

## First completed run

The five-seed pilot completed in 37.1 seconds on this laptop. It used 70,660
parameters per neural model and about 630 MiB of peak process memory.

Final accuracy on unfamiliar examples averaged 37.4 percent using only current
examples, 77.6 percent with a 256-example replay memory, and 87.0 percent with
the full history. Replay met the predeclared retention and acquisition criterion
in four of five seeds. One seed missed the acquisition tolerance.

The separate linear reference reached 89.3 percent. It uses explicit machine
identity and a model suited to these linear rules. Its training procedure differs
from the neural comparison.

The verified report includes all seed results, cost accounting, and limits.
Reports and charts are written under `runs/`, which this repository does not
include.

All 10 invariant checks passed. Saved-model verification reproduced all 15 neural
models' final scores and checked 552,960 training-example presentations for
permitted data access. The separate pipeline check is retained under checks.

## Run

This laptop already has CPU PyTorch and NumPy. The project environment reuses
those packages and adds Matplotlib for charts.

    cd continual-learning-lab
    .venv/bin/python -m unittest discover -s tests -v
    .venv/bin/python run_experiment.py --config configs/pilot.json

The runner prints the new run directory. Replace RUN_DIRECTORY below with that
path:

    .venv/bin/python report.py RUN_DIRECTORY

The report command verifies saved evidence and reproduces final checkpoint
predictions before creating plots, CSV files, and a report in the run's report
directory. It refuses to overwrite an existing report. To create another
derived report, supply a new path with the output option.

For verification alone:

    .venv/bin/python verify_run.py RUN_DIRECTORY

For a prediction on eight measurements you choose:

    .venv/bin/python predict.py RUN_DIRECTORY --seed 11 --method replay --machine 0 --values 0.2 -0.3 0.8 0.1 -0.6 0.4 0.7 -0.2

This prediction command reports the chosen bin and the simulator's correct bin.
It does not train or modify the saved model.

## Set up elsewhere

Create an environment and install the requirements. The PyTorch installation
must match the operating system and hardware. This project uses only its CPU
operations.

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

The recorded environment in each run identifies the versions actually used.

## What to change

Copy the pilot configuration to a new file before changing a scientific setting.
The runner saves its own immutable copy and always creates a new run directory.

| Setting | Effect |
| --- | --- |
| memory_capacity | Number of old examples retained by replay; zero disables storage |
| replay_fraction | Fraction of each batch drawn from the old memory |
| hidden_width | Number of units in each hidden layer |
| steps_per_task | Training budget for each newly introduced machine |
| batch_size | Number of example presentations per optimizer update |
| train_examples | Number of distinct training examples available per machine |
| tasks | Number of machines introduced |
| learning_rate | Size of optimizer updates |
| seeds | Independent model, data, and task-order starting states |
| cpu_threads | CPU threads used by PyTorch |
| wall_seconds_limit | Time limit for one scored run |

The curve_steps list must start at zero and end at steps_per_task.
Decision thresholds must be chosen before examining a new run's results.

## Files

- continual_lab.py defines data generation, the model, memory, and learning methods.
- run_experiment.py freezes inputs and runs the comparisons.
- verify_run.py checks evidence and checkpoint predictions.
- report.py creates charts and readable results.
- predict.py lets you inspect individual trained predictions.
- memory_sweep.py runs the fixed memory-size study and conditional confirmation.
- analyze_memory_sweep.py selects using validation data and reports the study.
- replay_allocation.py compares replay shares at fixed memory and training budget.
- analyze_replay_allocation.py audits actual replay counts and reports the study.
- adaptive_controller.py defines the fixed feedback rule and records its probes.
- adaptive_training.py adds the controller while preserving the original training implementation.
- adaptive_study.py runs adaptive replay and the three paired references on fresh seeds.
- analyze_adaptive.py reproduces probe predictions and reports accuracy and compute costs.
- tests/test_experiment.py checks the experiment's main invariants.
- tests/test_memory_sweep.py checks paired configurations, fresh seeds, and selection.
- tests/test_replay_allocation.py checks share selection and catches incorrect batch allocations.
- tests/test_adaptive_replay.py checks baseline equivalence, held-out-data isolation, and controller audits.

## Limits

The rules are synthetic and have linear boundaries. Machine identity is given.
The task provides correct labels, and each new training block arrives all at
once. This is supervised learning in stages, not autonomous exploration.

The experiment can demonstrate learning and forgetting in this setting. It
cannot establish language ability, human-like learning, or a replacement for
large-model pre-training.
