# SVM comparison

Exploratory comparison on previously inspected subjects 5-6. Not a new independent final test.

Same UP-Fall belt subset, 12 features and causal one-second windows as the original models.
StandardScaler was fit on subjects 1-3 only. Settings were selected on subject 4, then frozen.

| Reused subjects 5-6 | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| decision_tree | 95.88% | 37.40% | 82.14% | 0.514 |
| random_forest | 97.30% | 49.21% | 55.36% | 0.521 |
| svm | 89.01% | 14.23% | 62.50% | 0.232 |

Confusion matrices in svm_report.json use [[TN, FP], [FN, TP]].

| Method | Detected fall intervals | False groups in non-fall recordings |
|---|---:|---:|
| decision_tree | 10/10 | 15 |
| random_forest | 10/10 | 4 |
| svm | 10/10 | 31 |

Event counts use the original one-to-one matching rule and short scored exposure, before cancellation.
A high event recall can coexist with missed fall windows; they are different measurements.

## Selected settings

```json
{
  "C": 10.0,
  "kernel": "rbf",
  "gamma": 0.1,
  "class_weight": "balanced",
  "threshold": 0.35,
  "support_vectors": 550
}
```

## Limitations

- Exploratory comparison on previously inspected subjects 5-6. Not a new independent final test.
- Only six participants and one trial per activity; overlapping windows are not independent events.
- Laboratory belt data; no live hardware or construction-worker validation.
- Unknown activity tags and unsuitable timing windows excluded by the original policy.
- SVM scores are sigmoid-transformed margins, not calibrated probabilities or comparable confidence across models.
- Accuracy alone can be misleading: only 56 of the 2,111 comparison windows are falls.
- No settings were changed after this comparison. New participants are needed for a fresh final test.
- Wearable memory, latency and energy use have not been measured.

## Source and documentation

- UP-Fall dataset: https://doi.org/10.3390/s19091988
- SVM documentation: https://scikit-learn.org/stable/modules/svm.html
- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
