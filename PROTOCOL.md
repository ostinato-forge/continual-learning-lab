# Learning and retention pilot

This protocol is written before the first scored pilot. It defines a small
supervised experiment. It is not a test of language ability or general intelligence.

## Question

At equal neural training budgets, does replaying a small memory preserve earlier
rules while keeping the ability to learn newly introduced rules?

The primary comparison is replay against current-task training.
The all-history method and a separate linear model are reference comparisons.

## Simulated world

There are six machines. Each receives eight numerical measurements between -1
and 1 and chooses one of four bins. Its rule uses the signs of two fixed,
orthogonal linear combinations of the measurements. Each machine has its own
rule and a permutation of the bin names.

The neural network receives the measurements and six binary indicators identifying
the machine. The rule coefficients and bin permutation are not network inputs.
The architecture and the supplied machine identity are explicit design choices.
There is no claim that the learner has no prior structure.

For each machine and seed, generate 512 training examples, 256 validation
examples, and 1,024 test examples. Inputs and example IDs are disjoint between
splits. The examples come from the same distribution. This tests new examples,
not transfer to a new rule family.

The machine order is shuffled for each seed. A machine's training examples become
available together when that machine is introduced. The model receives correct
labels from the simulator. It does not choose its own observations.

## Neural methods

All methods start from the same randomly initialized weights within each seed.
The model has two hidden layers, each with 256 units, and a four-class output.
There is no pretrained model or pretrained representation.

Each machine gets 96 optimizer updates with 64 examples per update. The
optimizer is AdamW with learning rate 0.002 and weight decay 0.0001. Optimizer
state continues across machine blocks for all methods.

1. All earlier examples: sample uniformly from the current and all earlier
   training blocks. The method retains the complete history.
2. Current examples: sample only from the current machine. Discard access to
   earlier examples when the next machine arrives.
3. Small replay memory: use 32 current examples and 32 examples from a memory
   of earlier machines. Use 64 current examples when the memory is empty.
   After a block, add its examples using uniform reservoir sampling, which
   maintains a random sample of all earlier training examples. Capacity is 256.

Sampling is with replacement. Equal update counts and batch sizes mean equal
total example presentations, not equal presentations of new examples. Replay
spends some of its budget on old examples. Available examples, actually sampled
examples, and retained examples are reported separately.

The all-history method has more memory and a different allocation of its updates
across tasks. It is not an optimized upper bound.

## Simple reference

Fit a separate linear classifier for each known machine by ridge regression,
which fits linear scores while discouraging large coefficients. It uses all
training examples for that machine. The model receives machine identity and
never shares learned parameters between machines.

This reference has 216 fitted parameters in total. It uses a different learning
procedure and is not matched to the neural update budget. It tests whether a
simple method suited to the generated rules already performs well.

## Fixed measurements

Use seeds 11, 23, 37, 53, and 71. Rotate neural method execution order across
seeds to reduce systematic timing bias. Use two CPU threads.

After each machine block, measure test accuracy on all machines. Scores for
future machines are retained for inspection but never used for training or
configuration changes. Mean seen-task accuracy includes only introduced machines.

Record validation accuracy on the current machine at updates 0, 1, 4, 16, 32,
64, and 96. The exploratory target is 80 percent. A first crossing is only the
first recorded crossing. It is not the minimum sample count needed for learning.

Primary metrics:

- Final mean test accuracy across the six machines.
- Mean acquisition accuracy: accuracy on each machine immediately after its block.
- Earlier accuracy lost: acquisition accuracy minus final accuracy, averaged
  over the first five machines. This is signed; negative values indicate improvement.
- Optimizer updates, total example presentations, and retained-example count.
- Training, evaluation, and memory-maintenance time.

The pilot supports the stated replay hypothesis if at least four of the five
seeds show both:

- At least five percentage points less earlier accuracy lost than current-task
  training.
- No more than five percentage points lower mean acquisition accuracy.

This is a predeclared descriptive pilot criterion. It is not a significance test,
an estimate of performance on arbitrary tasks, or evidence of a new algorithm.

## Evidence and stop rules

Every scored invocation creates a new directory. Before training, save the
configuration, source files, environment versions, all generated data, and SHA-256
hashes. Save every sampled training ID, per-stage results, checkpoints, and an
append-only event log.

No tuning on the pilot test results is allowed within this run. A later change
requires a new run and an explanation. Preserve failed, incomplete, and negative
outcomes. Do not overwrite them.

Stop on non-finite loss, data-access violations, inconsistent budgets, or the
600-second run limit. A timeout is incomplete. Library import and later report
generation are outside this run limit.

Verification must check file hashes, paired initial weights, equal neural update
budgets, every sampled ID against the examples the method could access at that
stage, and final predictions loaded back from each checkpoint.

The complete benchmark process stores the simulated world and evaluation data.
Its peak process memory is not per-method deployment memory. Logical retained
data sizes are separate fields.

## Sources

Replay is an existing method:
[On Tiny Episodic Memories in Continual Learning](https://arxiv.org/abs/1902.10486).

The broader direction includes adaptation during use:
[Test-Time Training on Nearest Neighbors for Large Language Models](https://arxiv.org/abs/2305.18466).

This pilot does not reproduce either paper's benchmarks.
