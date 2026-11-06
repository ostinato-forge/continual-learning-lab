# Task partition study

Test whether separating some or all task parameters reduces forgetting at a
fixed parameter budget. This plan is fixed before scoring. Earlier failed
criteria remain unchanged.

## Architectures and controls

Run five conditions on the same data and task order within each seed:

- The original shared network with current-only learning.
- The original shared network with fixed 50 percent replay and 1,024 stored examples.
- The original shared network with full training history.
- Two shared hidden layers with a separate output layer for each task, trained
  on current examples only. Hidden widths are 245 and 248.
- Six independent small networks, trained on current examples only. Each uses
  eight raw features and hidden widths of 96 and 108.

The original shared network has 70,660 trainable parameters. Separate output
layers give 70,659 total parameters. Independent networks give 70,656 total
parameters, with 11,776 per task. All are within four parameters of the same
70,660-parameter ceiling. No unused padding parameters are added. The widths
were selected from parameter counts alone, before examining scored data.

The benchmark supplies task identity to all methods. The original network and
shared hidden layers receive it as input. Separate output layers and independent
networks also use it to choose which task component runs. This is supplied
routing, not learned task recognition. All six task components are allocated
at initialization and counted in the budget, including untrained components.

Shared features can still change when using separate output layers. Fully
independent networks have no shared learned weights. Their earlier predictions
should stay fixed when another task is learned. This expected protection is
an architecture property; it is not evidence of transfer between tasks.

## Learning and costs

Keep the original optimizer settings, 96 updates per task, batch size 64, and
576 total updates. Every model gets 36,864 training-example presentations.
New architectures store no old training examples and use no replay or probes.
Their inactive task parameters must receive no gradients and remain unchanged,
including after AdamW optimizer steps. AdamW is the existing optimizer, which
can change parameters through both gradients and weight decay.

The shared reference methods have identical starting weights within a seed.
Different architecture shapes cannot share identical weight tensors. They use
the same seed and the same layer initialization procedure, but their initial
weights are distinct. Within the three current-only conditions, batch IDs must
match exactly to isolate architecture from example exposure.

Record total and active parameter counts, training time, evaluation time, memory
work, and evidence capture time. Also report dense-layer multiply-add operations
per example in a forward pass. This operation count excludes routing, activation,
optimizer, and backward work. Equal parameter and update budgets do not imply
equal arithmetic or wall time. Timing remains subject to system load.

## Fixed decision and fresh seeds

Use seeds 1231, 1259, 1283, 1307, 1321, 1361, 1399, 1423, 1451, and 1483.
Check that they do not appear in earlier scored run configurations.
The older reserved confirmation seeds remain unused.

Both new architectures are fixed in advance. Test both directly on fresh test
data. There is no selection followed by a claimed independent confirmation.
For each architecture, require all three conditions in at least eight of ten
seeds, using the original shared current-only model as the paired reference:

- Final mean test accuracy is at least 80 percent.
- Earlier accuracy lost is at least five percentage points lower.
- Mean acquisition accuracy is at most five points lower. Acquisition means
  accuracy just after learning each new task.

Apply the same rule to fixed replay. Report every failure with unrounded
decisions. Compare each new architecture with fixed replay descriptively.
The seed threshold is a working criterion, not a significance test. Do not
tune architectures or replace seeds after scoring.

## Evidence

Freeze plan, source, configuration, and generated data before training. Keep
all scored, failed, incomplete, and negative results. Every run gets a new path.
The original baseline implementation and earlier saved studies stay unchanged.

Save batch IDs, final checkpoints, and a model state after each task for the new
architectures. Record hashes of shared and task-specific components before
learning and after each task. Audit actual parameter counts, inactive parameter
protection, all stage predictions, training access, and identical current-only
batch traces. Changing held-out labels must not affect training in software checks.

The study trains 50 neural models under a 7,200-second wall-clock limit.
Final verification and reporting happen afterward. A timeout is incomplete.
First exercise the pipeline on separate tiny data, without treating those
scores as research.

Before scoring on November 2, the user requested resource throttling because
earlier runs interfered with other work. All conditions now use one PyTorch
thread. The launcher pins the job to one CPU, gives it low scheduling priority,
and allows it to run for half of each 200-millisecond interval. Other numerical
libraries also receive a one-thread limit. This targets at most half of one
CPU on average, with short bursts limited to one CPU.

Each child has a hard 2 GiB address-space limit. A watchdog stops the process
group if sampled resident memory exceeds 1 GiB or available system memory falls
below 4 GiB. Resident memory is checked about every 200 milliseconds, so that
guard can briefly overshoot. Only one limited lab job can run at a time.

The time allowance increased from 900 to 7,200 seconds to allow throttling.
Architecture, seeds, update counts, data, and success thresholds did not change.
The recovery snapshot preserves the earlier plan. Resource limits and sampled
usage are saved. Wall times from this study are not comparable with the earlier
unthrottled studies.

## Interpretation limits

These are synthetic supervised tasks with linear rule boundaries and supplied
task identities. Separate networks may be a good fit for these independent
rules. Keep the simple separate linear reference to show that limitation.

The study does not test unknown task boundaries, language, shared concept
learning, transfer to new task families, or an unlimited stream of tasks.
Adding more independent task slots would require another capacity decision.

## Commands

    python3 limited_run.py .venv/bin/python partition_study.py --plan configs/partition_study.json
    python3 limited_run.py .venv/bin/python analyze_partition.py STUDY_DIRECTORY
