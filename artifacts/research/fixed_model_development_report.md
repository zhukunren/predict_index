# Fixed Candidate Development Study

The development results support carrying the predeclared ExtraTrees model into
one independent comparison. They do not establish that replacing the existing
production algorithm will preserve its predictive accuracy.

## Contract

- Signal dates: 2023-01-01 through 2024-12-31, 484 completed daily predictions.
- Held-out signal dates: 2025-01-01 onward. Their performance was not calculated.
- Target: next close divided by current close minus one, classified as up only
  when strictly positive. All neutral returns remain in the sample.
- Information time: after the signal-day close. Each fitting slice ends before
  the current signal row, so its last label is known at that close.
- Training: trailing 756 rows, minimum 252 rows, refit when normalized row index
  is divisible by 5 or a model has not yet been fitted.
- Features: existing core OHLCV features and permitted lagged Hang Seng inputs.
  Existing forward-fill and zero initialization are used; no backward fill.
- Direction threshold: 0.5, with no threshold search or class reweighting.
- Three candidates were declared before execution. Each was tried once without
  parameter changes. No ensemble or adaptive model selector was tested.
- The first following close resolves the last 2024 signal. No signal dated in
  2025 or later is included in any metric.

The JSON contract records full feature names, parameters, and source/input
hashes. The attempts JSON records each execution and its fitting count.

The research run starts fitting at its first development signal, whereas the
production implementation starts at historical row 252. In the current input,
the first development row is 728 (2023-01-03): research fits at row 728 while
production retains its row-725 model. Predictions on 2023-01-03 and 2023-01-04
can therefore differ. Both implementations refit at row 730 (2023-01-05) and
use the same schedule thereafter. The figures below describe the research
schedule, not an exact replay of production on those two initial dates.

## Results

| Model | 2023 Accuracy | 2024 Accuracy | Combined Accuracy | Combined Brier |
| --- | ---: | ---: | ---: | ---: |
| Logistic L2 | 51.65% | 56.61% | 54.13% (262/484) | 0.247378 |
| HistGradientBoosting | 49.17% | 58.26% | 53.72% (260/484) | 0.251354 |
| ExtraTrees | 54.96% | 57.02% | 55.99% (271/484) | 0.248283 |
| Always up | 48.35% | 51.24% | 49.79% (241/484) | Not a probability baseline |
| Training up-rate prior | 49.17% | 48.76% | 48.97% (237/484) | 0.250425 |

Always-up predictions are assigned probability 1 in the machine-readable output
to keep their direction consistent. Their Brier score should not be used as a
competitive probability baseline; the training up-rate prior and constant 0.5
(Brier 0.25) are the useful probability references.

## Frozen Recommendation

ExtraTrees has the highest combined direction accuracy and exceeds always-up in
both development years. Freeze these parameters before examining held-out data:

```python
ExtraTreesClassifier(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=30,
    max_features=0.7,
    n_jobs=1,
    random_state=42,
)
```

Logistic L2 has the lowest development Brier score and is the simpler probability
model. The reported ExtraTrees probabilities are native model scores, not
independently calibrated estimates of accuracy. All probability improvements
over 0.5 are small. Accuracy was the stated selection criterion; selecting a
different candidate after seeing held-out results would require a new untouched
evaluation period.

Before replacement, compare the frozen candidate against the original pipeline
and a fixed always-up baseline on identical held-out dates and labels. Preserve
the original pipeline when the predeclared non-regression condition fails. The
development advantage alone cannot guarantee future performance.

## Reproduction

```powershell
python -B artifacts/research/fixed_model_development.py
```

The experiment does not retrain the neural network, change core source code,
write production predictions, or use the network. It generated the development
contract, attempt log, daily predictions, and aggregate summary in this folder.
