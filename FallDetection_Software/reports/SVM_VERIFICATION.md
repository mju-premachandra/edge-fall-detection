# SVM update verification

Verified on 20 October 2026 in the build environment.

## Training and saved-model checks

- Nine SVM configurations fitted on training subjects 1-3. Each considered three score thresholds on validation subject 4.
- StandardScaler fitted on 3,237 training windows only; the notebook checks its fitted mean against the training data.
- Selected C=10, RBF gamma=0.1, class_weight=balanced and display-score threshold=0.35. No selection settings changed after comparison.
- Selected model exported before loading reused comparison subjects 5-6.
- Portable SVM decision margins match scikit-learn on all 6,495 train, validation and reused-comparison windows.
- Maximum absolute margin difference: 3.56e-12; required tolerance 1e-9.
- Eight notebook code cells executed in one Python namespace, capturing real output. This was not execution through a Jupyter kernel.
- Original feature extraction, Decision Tree, Random Forest and their reports remain byte-for-byte unchanged.

## Application checks

26 automated tests passed, including existing tests and seven new SVM tests.
Checks cover actual selected-model inference across recording, JSON, pasted CSV and uploaded CSV inputs; labels excluded from features; score stability; invalid model scaling; cancellation, expiry, acknowledgement, persistent model identity and legacy database migration.

## Browser checks

Chromium Headless 140; Playwright 1.55

- Three model options load from the backend
- SVM selection displays its actual evaluation and score explanation
- Known walking excerpt runs through the SVM and predicts non-fall
- SVM fall input opens a cancellation window and cancellation succeeds
- SVM alert expires and the local receiver acknowledges it
- CSV upload uses selected SVM
- Switching clears old predictions and both existing models still run
- SVM event history and model identity survive page reload
- 390px mobile layout has no horizontal overflow
- No JavaScript page errors. Desktop and mobile screenshots visually inspected.

A clean installation test also overlaid this add-on onto the original base ZIP without the Random Forest add-on. Both SVM walking and fall examples passed.

## Scope and limitations

- Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5 and pandas 2.2.3 in the build environment.
- Windows launcher reviewed; Windows launch and the user's Python 3.13 environment were not executed here.
- Reused comparison data: not a fresh independent evaluation. SVM has lower F1 and more false alarms than the existing models on this subset.
- Saved models and reports load at server startup; restart after retraining.
- Hardware behavior, mobile packaging, energy use and live deployment are not tested.
