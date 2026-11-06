# Memory capacity study

This plan is fixed before inspecting the new study's results.

The original pilot is retained at runs/20261031T181522Z-17e36c26.
Its five seeds are already observed evidence and are not used here.

## Questions

1. Does the original 256-example replay result hold on fresh seeds?
2. What is the smallest tested memory that meets a useful accuracy and retention
   target while preserving new learning?
3. Does the selected memory meet that target on additional, untouched seeds?

## Fixed settings

Use the pilot's learning code, task distribution, architecture, optimizer,
96 updates per task, 64 presentations per update, and 50 percent replay fraction.
Keep all other scientific settings fixed.

Compare capacities of 0, 16, 64, 128, 256, 512, and 1,024 examples. Capacity zero
uses the current-only method. Run a full-history reference once per seed.
Retain the separate linear reference.

Use exploration seeds 101, 127, 149, 173, 197, 223, 251, 277, 307, and 337.
Within each seed, all capacities use identical data, task order, and initial
neural weights. These matches must be verified from saved artifacts.

The new wrappers call the existing single-run implementation. They do not change
the learning algorithm. Individual replay-only runs do not make a paired
decision in isolation. The study combines them with the shared current-only
reference after verification.

## Selection rule

Select using validation scores only. Exploration test scores may be reported,
but they must not enter the capacity selection function.

A capacity qualifies if at least eight of ten exploration seeds satisfy all
three conditions:

- Final validation accuracy is at least 80 percent.
- Earlier validation accuracy lost is at least five percentage points lower
  than current-only training.
- Mean new-task validation acquisition accuracy is no more than five percentage
  points lower than current-only training.

Choose the smallest tested qualifying capacity. The 80 percent threshold is a
working target for this study. It is not a universal definition of useful learning.

If none qualifies, report that outcome and skip confirmation. Do not relax the
target after seeing the results.

## Replication of the original comparison

Separately apply the original retention and acquisition criterion to the
256-example method using the ten exploration seeds' test scores.
Require at least eight successes.

This check does not include the new 80 percent final-accuracy target.
Its outcome does not change the capacity selected from validation data.

## Fresh confirmation

If a capacity qualifies, test it with seeds 401, 433, 463, 499, and 541.
Run current-only, selected replay, and full-history methods on those seeds.

The selection file must be written and hashed before confirmation starts.
The confirmation seeds are already fixed by this plan.

Apply the same three selection conditions to held-out test scores. Require
success in at least four of the five confirmation seeds.
Report individual failures and the mean scores. If confirmation fails, retain
the failure and do not choose a replacement capacity within this study.

## Evidence and cost

Freeze this plan, source, base configuration, and all exploration configurations
before training. Every condition creates a separate run with its own frozen
datasets, source, batch traces, checkpoints, and metrics.

Verify every condition using the existing verifier. Also check that datasets,
task order, initial weights, and training budgets match across capacities.
The report must reproduce the declared selection from the saved validation
metrics and verify that no artifacts changed after selection.

The exploration stage trains 80 neural models. Confirmation adds 15 if a
capacity qualifies. Each model gets the same 576 updates and 36,864 example
presentations. The complete training and selection study has a 600-second
limit. Final report generation and confirmation readback occur afterward.

Retain failed, incomplete, and negative runs. The pipeline check uses separate
small data and explicitly relaxed thresholds solely to exercise the code path.
It is not scientific evidence for either hypothesis.

## Interpretation limits

This remains one synthetic, supervised task family. Machine identity is supplied,
and the rules have linear boundaries. Memory size counts retained training
examples, not total process memory or all information inherited by the system.

The smallest tested qualifying capacity is not necessarily the true minimum.
The seed counts and thresholds are descriptive experimental choices, not a
statistical significance test or a guarantee on other tasks.

## Commands

    .venv/bin/python memory_sweep.py --plan configs/memory_sweep.json
    .venv/bin/python analyze_memory_sweep.py report STUDY_DIRECTORY
