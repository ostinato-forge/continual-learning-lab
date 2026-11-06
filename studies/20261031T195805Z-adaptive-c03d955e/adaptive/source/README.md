# Continual learning lab

Run a small neural model on a sequence of new rules. Compare learning from current
examples, replaying a small memory, and retaining all earlier examples.

The first experiment asks whether memory replay reduces forgetting at a fixed
training budget. It starts from random weights and uses generated numerical
inputs with labels supplied by a simulator.

Read [PROTOCOL.md](PROTOCOL.md) for the rules fixed before the first scored run.
The main configuration is [configs/pilot.json](configs/pilot.json).

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

Read the [verified report](runs/20261031T181522Z-17e36c26/report/REPORT.md) or
view the [charts](runs/20261031T181522Z-17e36c26/report/results.png).
The report includes all seed results, cost accounting, and limits.

All 10 invariant checks passed. Saved-model verification reproduced all 15 neural
models' final scores and checked 552,960 training-example presentations for
permitted data access. The separate pipeline check is retained under checks.

## Run

This laptop already has CPU PyTorch and NumPy. The project environment reuses
those packages and adds Matplotlib for charts.

    cd /home/eparenti/continual-learning-lab
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
- tests/test_experiment.py checks the experiment's main invariants.
- tests/test_memory_sweep.py checks paired configurations, fresh seeds, and selection.
- tests/test_replay_allocation.py checks share selection and catches incorrect batch allocations.

## Limits

The rules are synthetic and have linear boundaries. Machine identity is given.
The task provides correct labels, and each new training block arrives all at
once. This is supervised learning in stages, not autonomous exploration.

The experiment can demonstrate learning and forgetting in this setting. It
cannot establish language ability, human-like learning, or a replacement for
large-model pre-training.
