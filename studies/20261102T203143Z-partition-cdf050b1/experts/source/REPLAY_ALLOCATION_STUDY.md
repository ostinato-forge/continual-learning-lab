# Replay allocation study

This plan is fixed before the first scored allocation run. It follows the
negative memory-size result without changing or replacing that evidence.

## Question

Can a smaller replay share preserve new learning while retaining enough earlier
learning? The previous study used half of each later batch for old examples.
At 1,024 stored examples, final accuracy and retention passed every validation
seed, but the new-learning tolerance passed only seven of ten.

Hold replay memory at 1,024 examples. Compare replay shares of 12.5, 25, and
50 percent. Add current-only training and the full-history reference.
The original model, task generator, optimizer, and training code stay fixed.
Each neural model has 70,660 parameters and receives 576 updates, with 64
examples per update. Changing the replay share changes which examples receive
those updates. It does not reduce the total update budget.

After the first task, the three replay conditions draw 8, 16, or 32 old examples
per batch. The first task draws only current examples because memory is empty.
Memory uses uniform reservoir sampling and receives the current block only
after that block has been learned.

## Exploration and selection

Exploration seeds: 601, 631, 661, 691, 719, 751, 787, 821, 853, and 887.
The prior pilot and memory-study seeds are excluded. Within each seed, all
conditions must share data, task order, and starting weights.

Selection uses validation metrics only. Test metrics may be reported afterward
but cannot enter the selection function. A share qualifies when all three
conditions hold in at least eight of ten exploration seeds:

- Final validation accuracy is at least 80 percent.
- Earlier validation accuracy lost is at least five percentage points lower
  than current-only training.
- Mean new-task validation acquisition accuracy is at most five points lower
  than current-only training. Acquisition means accuracy immediately after
  learning each new task.

Choose the smallest positive tested qualifying share. If none qualifies,
report the negative result and skip confirmation. Do not relax the target.

## Fresh confirmation and reference

Confirmation seeds: 401, 433, 463, 499, and 541. These were reserved in the
memory-size plan but were never used because that study selected no capacity.
Their absence from prior scored run configurations was checked before this plan.

Write and hash the selection before confirmation starts. Run current-only,
full-history, and the selected replay share on these five seeds. Apply the same
three conditions to test metrics. Require all three in at least four of five
seeds. Report every failure. A failed confirmation does not permit selecting a
replacement share within this study.

If the selected share differs from 50 percent, also run a 50 percent replay
reference on the same confirmation data and starting weights. Report paired
differences in acquisition, final accuracy, and earlier accuracy lost.
These differences are descriptive. They do not change the acceptance criterion
or establish statistical significance, general superiority, or cheaper compute.
If 50 percent is selected, the selected method itself supplies this reference.

## Evidence and cost

Freeze source, plan, base settings, and exploration configurations before
training. Preserve all datasets, batch IDs, checkpoints, scores, and failures.
Check each final checkpoint, all permitted training access, and the actual
number of replay examples in every batch. Check pairing across conditions.

Exploration trains 50 neural models. Confirmation adds 15, plus five paired
reference models if the selected share is below 50 percent. Each model gets
36,864 example presentations. The training and selection study has a 600-second
limit. Final confirmation readback and reporting occur afterward.

Before scoring, exercise the complete pipeline on a tiny separate dataset with
relaxed thresholds. That pipeline check has no scientific interpretation.
Do not tune the scored settings from its output.

## Limits

The task family is synthetic and supervised. Machine identity is supplied, and
the rules have linear boundaries. The separate linear reference uses a model
suited to these rules and a different learning procedure. This is not a test of
language learning or replacement of large-model pre-training.

The acceptance thresholds are fixed experimental choices, not a statistical
significance test. Selecting the smallest tested qualifying share does not find
a universal optimum. Prior negative outcomes remain part of the evidence.

## Commands

    .venv/bin/python replay_allocation.py --plan configs/replay_allocation.json
    .venv/bin/python analyze_replay_allocation.py report STUDY_DIRECTORY
