# Random Forest comparison

Exploratory comparison on previously inspected subjects 5-6. Not a new independent final test.

Same six-axis belt data, 12 features and causal windows as the original model.
Training subjects 1-3; settings selected using subject 4 only.

| Reused subjects 5-6 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| decision_tree | 95.88% | 37.40% | 82.14% | 0.514 |
| random_forest | 97.30% | 49.21% | 55.36% | 0.521 |

## Event comparison (before cancellation)

| Method | Detected tagged fall intervals | False groups in non-fall recordings |
|---|---:|---:|
| decision_tree | 10/10 | 15 |
| random_forest | 10/10 | 4 |

These event counts use the original matching rule and short scored exposure; they do not establish an operational false-alarm rate.

## Selected forest

{
  "n_estimators": 100,
  "max_depth": 5,
  "min_samples_leaf": 5,
  "threshold": 0.65
}

## Limits

- Exploratory comparison on previously inspected subjects 5-6. Not a new independent final test.
- Only six participants, one trial per activity, and overlapping windows; not independent events.
- Laboratory belt data, not construction-worker or live-hardware validation.
- Unknown labels and unsuitable timing windows excluded by the original policy.
- A forest score is not a calibrated probability; accuracy alone is misleading on this imbalanced subset.
- No architecture or threshold was adjusted after this comparison evaluation.
- Model size and actual wearable deployment need separate assessment.

The current Flask frontend still uses the original decision tree. Use random_forest.py or the notebook to try the second model.
