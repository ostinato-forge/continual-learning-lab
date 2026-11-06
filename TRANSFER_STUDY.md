# Shared feature transfer study

Test whether earlier supervised learning helps with new combinations of known
hidden rules. This protocol and configs/transfer_study.json are fixed before
scoring. Earlier experiments, negative results, and reserved seeds stay intact.

The primary question concerns labelled examples needed for later learning.
Initial learning is included in the cost report. A reduction in later examples
does not by itself establish lower total computation.

## Tasks and information

Keep eight raw numerical inputs, four classes, explicit task identity, uniform
inputs in [-1, 1], and supervised task blocks. Generate four orthonormal hidden
directions per seed. A task uses the signs of two projections to define four
classes, followed by a fixed random permutation of the class names.

The four source tasks use pairs (0, 1), (1, 2), (2, 3), and (3, 0). The two
target tasks use the held-out pairs (0, 2) and (1, 3). Each direction appears in
source tasks, but neither target pair does. Randomize order within the source
block and within the target block. Targets always follow all sources.

The learner receives raw inputs, task identity, and class labels. Hidden
directions and pair definitions are generator evidence, not learner inputs.
There is no pretrained model or external dataset. This changes the earlier
benchmark from independent rules to shared rules. It does not test discovery
of task boundaries or unknown task identity.

Use 512 training, 256 validation, and 1,024 test examples per task. Splits have
distinct identifiers and no identical raw rows. Target training uses nested
prefixes of 32, 128, or 512 examples. All points are fixed in advance and are
reported. There is no selection of a winning sample size. Validation scores
are diagnostic only and cannot affect updates, stopping, or model choice.

## Methods

All shared models have two hidden ReLU layers of width 248 and six separate
four-class output layers. The hidden layers receive only the eight raw inputs.
Supplied task identity selects the output layer. Total parameters are 69,960.
Independent networks use widths 96 and 107, with 70,050 parameters across six
networks. Both are below the 70,660 ceiling and differ by 90 parameters. Count
all allocated components, including ones not yet trained.

1. Learned frozen features: learn the four sources with fixed 50 percent
   replay and at most 1,024 stored examples. Then freeze both hidden layers,
   discard replay memory, and train only each target's output layer.
2. Random frozen features: freeze the identical initial hidden layers without
   source learning. Train target output layers on the same examples and batch
   draws as learned frozen features. This controls for useful random features.
3. Fresh model: initialize the same shared architecture separately for each
   target and train its hidden layers and current output layer. It is a
   target-learning reference and does not retain the four source tasks.
4. Fixed replay: start from the identical learned source checkpoint and
   continue updating hidden and output layers with 50 percent replay. Retain
   at most 1,024 old training examples.
5. Independent networks: learn one separate network per task. Source networks
   and earlier target networks remain unchanged. This is the bounded-state
   reference from the preceding study, adjusted only to the declared widths.
6. Simple reference: fit a separate ridge linear classifier from raw inputs
   and an intercept for each target training prefix. Report its different
   fitting procedure and cost separately.

The shared source checkpoint is trained once per seed, then copied into each
learned-feature and replay branch. Charge its full cost once to every complete
source-plus-target scenario. Independent source training is handled the same
way. Sharing this work during the experiment reduces repeated execution; it
does not make the source learning free in a method's reported cost.

Each sample-size branch starts from its source checkpoint or identical random
initialization. It cannot receive weights, examples, or optimizer state from
another sample-size branch. The fresh reference resets for each target.

## Training and bounded state

Use AdamW with learning rate 0.002 and weight decay 0.0001. Reset optimizer
state at each task for every method. This is declared separately from the
earlier study, whose optimizer persisted across tasks. The new method
comparisons all use the same reset rule.

Each source task and target task receives 96 updates of 64 examples. Sample
with replacement. Non-replay target methods use identical batch identifiers
within each seed and sample-size branch. Replay uses 32 current and 32 old
examples when old memory is available. Its first source task uses 64 current
examples. Record actual current and replay presentations separately.

Source training sees all 512 training examples per source. A target branch can
use only its declared prefix. Reservoir updates use only permitted training
examples. Frozen methods and independent networks use no target replay.

Record parameter bytes, peak optimizer tensor bytes, replay examples and
bytes, available labels, unique sampled labels, update counts, and example
presentations. Keep evaluation data and saved checkpoints outside the learner's
retained-memory accounting and report that boundary. These are finite six-task
experiments, not constant-memory claims for an unlimited stream.

Report source training, target training, memory operations, evaluation, and
evidence capture separately. Use process CPU time as well as wall time under
the resource limiter. Count dense-layer multiply-add operations for forward
and backward matrix products as an arithmetic estimate. This excludes biases,
activations, loss, optimizer operations, routing, and memory work. It is not an
energy or complete compute measurement. Frozen trunks still run a forward pass
for every example; do not cache their features in this study.

Fresh and random-feature controls have no source-learning cost and cannot
provide a full-stream retention comparison. Report their target costs, without
presenting them as methods that completed the same six-task job.

## Fixed decision

Use ten fresh seeds: 1601, 1613, 1627, 1657, 1663, 1693, 1721, 1733, 1747,
and 1759. Reject any seed already present in scored or incomplete study records.
The earlier reserved confirmation seeds 401, 433, 463, 499, and 541 stay unused.

For a seed to support the primary transfer criterion, all conditions must hold:

- Learned frozen features with 128 target examples reach at least 80 percent
  final test accuracy on each target.
- Their mean target accuracy is no more than two percentage points below the
  fresh model trained with 512 target examples.
- Their mean target accuracy exceeds the random-feature control at 128
  examples by at least five percentage points.
- Source predictions remain exactly unchanged after target learning.

Require success in at least eight of ten seeds. Decide from unrounded values
and show every failed condition. This is a working criterion, not a significance
test. The 128 versus 512 comparison tests 75 percent fewer available target
training labels, not 75 percent fewer total labels or updates. Count source
labels and actual sampled labels too.

Compare fixed replay and independent networks at every sample size. Report
paired accuracy differences, forgetting, source cost, and complete-stream cost.
Do not select a replacement threshold if the primary criterion fails. Lower
total cost is a separate descriptive result and requires comparable delivered
accuracy and retention. Report the simple reference even if it wins.

## Evidence and resource controls

Freeze this protocol, the plan, source, generated data, environment, and resource
settings before training. Save source and target task-end checkpoints, exact
batch identifiers, permitted current identifiers, replay membership before and
after each task, component hashes, and outcomes. Keep every failure and incomplete
run. Each run and each report gets a new directory.

Software checks cover fresh seeds, nested sample sets, no held-out-data access,
source and target pairing, deterministic batches, source reuse, branch isolation,
frozen components, budget counts, decision boundaries, and tamper detection.
Verification reproduces checkpoint predictions and audits draw and memory
records. A tiny pipeline check uses separate seeds and settings. Its scores
are not research evidence.

Run one lab job at a time through limited_run.py with its unchanged defaults:
half of one CPU at low priority, one numerical-library thread, a 2 GiB address
space limit, a 1 GiB resident-memory watchdog, and a 4 GiB free-memory reserve.
The scored study has a 7,200-second wall limit. A limit stop is incomplete.
Never raise limits or silently repeat scored seeds to replace a result.

## Interpretation

Successful transfer would apply to a deliberately related synthetic task
family. It would not establish a novel learning algorithm, language learning,
unlimited continual learning, or an alternative to large-model pre-training.

Frozen-feature reuse and transfer through earlier representations are established
research directions. [Progressive Neural Networks](https://arxiv.org/abs/1606.04671)
uses connections to earlier features, with a different architecture. Research
on [continually learning representations](https://proceedings.mlr.press/v232/galashov23a.html)
also cautions that benchmark overlap can make frozen representations effective.
This study tests one explicit shared-rule construction and counts its initial
learning cost.
