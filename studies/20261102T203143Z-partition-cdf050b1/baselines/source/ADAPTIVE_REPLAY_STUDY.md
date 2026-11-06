# Adaptive replay study

Test whether a fixed feedback rule can improve the balance between new learning
and retention. This plan is fixed before scored runs. Earlier failed criteria
remain unchanged.

## Controller

Memory holds up to 1,024 old training examples using the existing uniform
reservoir. The current training block enters memory only after learning ends.

At the start of each task after the first, choose 128 unique examples uniformly
from the retained memory. Keep this probe set unchanged through that task.
Its selection uses a separate random generator, seeded from the experiment seed.
These remain training examples and may also be sampled in training batches.
The probe copies do not add distinct examples beyond the retained memory.

Start at 25 percent replay. Measure probe accuracy before the first update and
every eight updates afterward, through update 88. The initial accuracy is the
fixed reference for that task. At each later measurement:

- If accuracy has dropped by at least five percentage points, move up one level.
- If the drop is at most one percentage point, move down one level.
- Otherwise, keep the current share.

The allowed levels are 12.5, 25, 50, and 75 percent. Stay within these bounds.
Reset the controller and reference at each new task. The gap between the two
thresholds reduces switching when scores change slightly. The first task uses
only current examples and performs no probes.

The controller cannot access validation or test examples, labels, or scores.
Probe evaluation does not change weights or optimizer state. It only chooses
the composition of later training batches.

## Comparison and fixed decision

Run current-only, full-history, fixed 50 percent replay, and adaptive replay.
Every neural model has the original architecture and gets 576 updates with
64 examples per update. All four methods share data, task order, and starting
weights within each seed. The existing baseline training code stays unchanged.

Use ten fresh seeds: 907, 929, 953, 977, 1009, 1031, 1051, 1087, 1109, and 1151.
There is one predefined controller and no parameter selection. This is a direct
test on fresh test data, not an exploration sweep followed by selection.
The five confirmation seeds reserved by earlier studies remain unused.

Keep the earlier three targets. Adaptive replay passes this study's working
criterion only if all three hold in at least eight of ten test seeds:

- Final mean test accuracy is at least 80 percent.
- Earlier accuracy lost is at least five points lower than current-only training.
- Mean new-task acquisition accuracy is at most five points lower than
  current-only training. Acquisition means accuracy just after learning a task.

Apply the same rule to fixed 50 percent replay. Report every seed and the paired
adaptive-minus-fixed differences in final accuracy, acquisition, and forgetting.
Those differences are descriptive, not a separate significance test. Passing
the working criterion does not by itself establish superiority to fixed replay.
Do not tune the controller, replace seeds, or change the rule after scoring.

## Cost and evidence

Equal training updates do not mean equal total compute. Each model gets 36,864
example presentations with forward and backward passes. Adaptive replay adds
60 probe calls, each with 128 forward-only evaluations, for 7,680 extra example
evaluations. There are no additional gradient updates from probes.

Record training time, controller time, benchmark evaluation time, memory work,
and evidence capture time separately. Also record the actual replay share for
every batch. Model readback for auditing and report generation are separate
from the learner's compute. Timing comparisons remain subject to system load.

Freeze the plan, source, data, and configurations before learning. Save batch
IDs, probe IDs, controller decisions, and model states at probe points. Verify
the actual probe predictions, decision rules, memory access, batch composition,
final checkpoints, and paired inputs. Check that changing held-out labels does
not change training or controller decisions in a separate software test.

The study runs 40 neural models with a 900-second training limit. This limit is
chosen before scoring because the last 50-model study took 424.5 seconds and
runtime varied across conditions. Final verification and reporting occur after
the training limit. A timeout is incomplete, not evidence for either outcome.

First run a tiny, separate pipeline check. Its results are software evidence
only. Retain all scored, negative, failed, and incomplete runs.

## Limits

Probe accuracy on training memory is not held-out generalization. A controller
may preserve those examples while doing poorly on unseen inputs. Test metrics
measure this difference. The tasks are synthetic, supervised, and have supplied
machine identities and linear boundaries. Success would not establish language
learning, general intelligence, or replacement of language-model pre-training.

## Commands

    .venv/bin/python adaptive_study.py --plan configs/adaptive_replay.json
    .venv/bin/python analyze_adaptive.py STUDY_DIRECTORY
