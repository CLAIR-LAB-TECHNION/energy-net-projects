# Competitive Energy Market RL Exercise

> **Prototype and AI disclosure**
>
> This exercise was designed and implemented largely with AI assistance. It is
> an experimental prototype, not polished teaching material. Visualizations,
> interactive explanations, extensive refactoring, and other classroom aids
> were deliberately left out. The code is more complicated and less readable
> than ideal; part of the exercise is seeing whether students can understand,
> use, and improve an imperfect EnergyNet-based system, which is something you will have to deal with during your project while working with human or AI partners.
> This is not a homework - the only requirement is that you submit a passing result, and while the person with the best result might receive a prize, the primary purpose is simply to check that you are ready to tackle a real project in the lab.

## Part I — The problem

A learning electricity supplier, called the **main ISO**, must earn money while
competing with another learning ISO and serving a battery-equipped household
controller, called the **PCS**.

Each ISO chooses, for every half-hour:

- the retail price charged to the PCS,
- the price paid for electricity exported by the PCS, and
- how much generation to dispatch in advance.

The PCS buys from the cheaper ISO and sells genuine excess energy to the ISO
offering the higher payment. It may shift energy through its battery, but it
cannot buy and immediately resell energy in the same half-hour.

Dispatch always costs money. Demand above dispatch requires more expensive
emergency energy. The PCS only bothers (or code-wise, is allowed) to sell excess energy only when an ISO offers at least
`0.06` per unit; the ISO may resell it externally for `0.12`, leaving at most
`0.06` profit per exported unit.

All three participants learn during training:

- the PCS and competitor begin from reusable checkpoints produced by much
  longer pretraining runs, then continue adapting during the experiment;
- the competitor is a serious learned market participant, not a deliberately
  bad policy;
- the main ISO begins from new random weights and must learn within the short
  experiment budget;
- updates made during one run are discarded, so later runs start from the same
  checkpoints.

Training uses earlier chronological data. Evaluation uses later dates once,
without further learning. The combined competitive training target is **five
CPU-minutes** using one compute thread. CPU time measures how much processing a
job actually receives rather than how long it waits on the clock. This lets you
launch several algorithm or hyperparameter experiments at once without shorter
budgets for jobs sharing the machine; parallel jobs may take longer in wall
time, but each receives the same compute target. Training stops after the
current environment step and reports the exact CPU and wall time used.

**Success has one definition:** the main ISO's total held-out profit must be
strictly greater than zero. Other reported metrics explain the result but do
not affect pass/fail.

> 🏠 **A small data-history note:** the household readings came from a London
> electricity dataset that Michael imported, normalized to smaller values, and no longer remembers the exact source of 🙂
> Similar smart-meter household datasets are widely available online, so the
> experiment can later be repeated with a clearly documented public source.

## Part II — Code and commands

Students mainly need:

- `competitive_alternating.py`: the EnergyNet subclass and competitive step;
- `run_competitive_alternating.py`: the short experiment entry point;
- `grading/submission_template.py`: the required submission interface.

The `internal/` folder contains accounting, preprocessing, parameters,
checkpoint loading, and evaluation plumbing that students do not need to study.
The core `energy_net/` package is not modified.

### Rebuild the reusable models

Only required when the environment, reward, observations, or checkpoints
change:

```powershell
python -m for_students.training.train_pcs
python -m for_students.training.train_competitor_iso
```

### Run the five-minute experiment

```powershell
python -m for_students.run_competitive_alternating
```

The competitor is a demanding reference point with substantially more prior
training. You do not need to outperform it: success means that your own ISO
earns positive held-out profit.

### Submit a policy

Copy `for_students/grading/submission_template.py` and implement:

```python
def create_policy(observation_space, action_space):
    return policy_with_a_predict_method
```

The returned object must implement:

```python
action, state = policy.predict(observation, deterministic=True)
```

Actions must have the exact ISO action shape, contain only finite values, and
remain within `[0, 1]`.

Evaluate a submission with:

```powershell
python -m for_students.grading.evaluate_submission path\to\submission.py
```

The evaluator loads fresh PCS and competitor checkpoints, runs the later
chronological test period once, prints the financial and energy diagnostics,
and passes the submission only when total held-out ISO profit is positive.

For real grading, the final test data must be stored outside the student
checkout; this local evaluator cannot hide repository files from hostile code.
