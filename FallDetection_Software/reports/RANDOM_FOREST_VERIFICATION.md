# Random Forest verification — 18 October 2026

- Six candidate forests fitted on 3,237 eligible training windows from subjects 1-3.
- Selected settings using 1,147 validation windows from subject 4 only.
- Fixed plan: 100 trees, depths 5/8/unlimited, minimum leaves 2/5, thresholds .35/.50/.65.
- Selected depth 5, minimum leaf 5, threshold .65.
- Evaluated 2,111 reused comparison windows from subjects 5-6 after the new settings were frozen.
- Exported JSON forest scores matched scikit-learn scores within 1e-12 on every train, validation and comparison window.
- All 8 notebook code cells executed sequentially as plain Python; saved outputs are captured from actual execution. This was not a Jupyter-kernel or VS Code UI test.
- 13 unittest cases passed, covering the original backend workflow and new forest inference/report integrity.
- The exported forest independently reproduced the comparison confusion matrix [[2023,32],[25,31]].
- Input labels and subject IDs cannot alter forest predictions; invalid short/duplicate-timestamp inputs are rejected.
- Known training walking and fall excerpts were successfully classified by the new model; these examples are demonstrations, not additional validation.
- app.py, core.py, train_model.py, model/trained_tree.json and reports/training_report.json match the original software ZIP byte-for-byte.

Environment: Python 3.12.14, scikit-learn 1.8.0, NumPy 2.3.5, pandas 2.2.3. Flask checks used Flask 3.1.3.
The Windows launcher was reviewed but not run on Windows. Hardware, mobile integration and wearable inference were not tested.
Subjects 5-6 had already been examined before this experiment; a fresh independent final test needs new participants.
