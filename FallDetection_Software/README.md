# Fall Detection Software Prototype

A local, runnable version of the Edge-Based Fall Detection and Haptic Alert
System component. It accepts timestamped belt accelerometer/gyroscope readings,
runs an actually trained decision tree, returns predictions to the frontend,
and demonstrates cancellation, local escalation, acknowledgement and saved events.

Start with **START_HERE.txt**. On Windows, extract the folder beside your existing
`.venv313`, then double-click **start_demo.bat**. The app runs at
http://127.0.0.1:5050. No Jupyter session is required. Internet is needed to
install packages; the app itself uses local assets and included recordings.

## What you can demonstrate

| Input | Processing | Output |
|---|---|---|
| Included UP-Fall recording | Flask API validates values, extracts causal features and runs the saved tree | Per-window fall scores, decisions and a signal chart |
| Uploaded Step 2 belt CSV | Same trained model and validation | Actual backend JSON and predictions |
| Pasted or edited sensor sequence | Same trained model; no activity label used | Updated predictions from your entered readings |
| Suspected-fall decision | Local backend opens an editable cancellation period | Cancelled event, or escalation followed by acknowledgement |

This is a **software-only demonstration**. The visual alert represents the
proposed haptic prompt. The receiving service is the local backend. Hardware
acquisition, physical vibration, on-device deployment, live timing and external
notification delivery remain future work. An entered worker ID is a demo label,
not an authenticated worker identity.

## Input format

CSV columns:

```csv
time_s,acc_x_g,acc_y_g,acc_z_g,gyro_x_deg_s,gyro_y_deg_s,gyro_z_deg_s
```

- `time_s`: strictly increasing elapsed seconds. An ISO `timestamp` column is
  accepted instead; the first timestamp becomes zero.
- Accelerometer axes: **g**, including gravity as in the source data.
- Gyroscope axes: **degrees per second**, using the source axis convention.
- Provide a sequence covering at least one second. A single set of six numbers
  is insufficient for this temporal model. The builder appends at 0.05 s, so
  21 readings cover one second.
- A window must contain at least 10 readings, span at least 0.8 s and have no gap
  greater than 0.25 s. Candidate endpoints start at one second and are separated
  by at least 0.25 s. The system skips unsuitable windows; it does not interpolate.
- Maximum 20,000 rows and 5 MB per upload. Labels, subject IDs and activity names
  are ignored during prediction. Raw two-header UP-Fall files must first be
  converted to the Step 2 format; processed files are included here.

Use `examples/walking_window.csv` and `examples/fall_window.csv`, or the frontend
load buttons. These are real subject 1 **training** excerpts chosen to explain
the workflow. They are not new validation results. Values typed by hand are
synthetic test input, not newly collected research data. Editing a value need
not flip the prediction; a tree changes its decision only when feature thresholds
are crossed. Future hardware must match units, axes and placement and requires
new validation.

## Trained model and evidence

- Data: 66 processed recordings, subjects 1–6, all 11 activities, repetition 1.
  Training uses subjects 1–3; validation uses subject 4; final testing uses 5–6.
  Participant splitting happened before extracting overlapping windows.
- Features: 12 sensor-derived features from the current and preceding second.
  No future values, activity tags, filenames or subject IDs enter the classifier.
- Algorithm: balanced decision tree, depth 5, minimum leaf size 5, selected
  threshold 0.65. Selection used validation F2, then precision and simpler settings.
  Model settings were frozen before the held-out test was read.
- Held-out window results: precision **37.4%**, recall **82.1%**, F1 **0.514** on
  2,111 windows (56 fall windows). Overlapping windows are not independent trials.
- Event matching detected 10/10 tagged fall intervals, but produced **15 false
  alert groups** in about 7.49 minutes of scored non-fall exposure, before
  cancellation. This is a small laboratory prototype with substantial false
  alarms, not evidence of readiness for construction sites.
- There are 107 unknown row labels and irregular gaps in the source subset.
  Labelled windows containing unknown tags and windows failing timing checks
  are excluded from evaluation. See the data-quality and training reports.

The model is already trained. `model/trained_tree.json` contains the fitted
tree, not hand-written fall examples. Backend inference reads this portable tree
without scikit-learn or pickle. The training script checked that its scores
match the fitted scikit-learn model on every evaluated window within 1e-12.
Scores are weighted leaf fractions, **not calibrated probabilities**.

To reproduce training, from this folder in the existing Python environment:

```bat
..\.venv313\Scripts\python.exe -m pip install -r requirements-training.txt
..\.venv313\Scripts\python.exe train_model.py
```

Retraining overwrites the model and reports; preserve the supplied results if
you need to show this exact evaluated version. Original training used Python
3.12.14, NumPy 2.3.5 and scikit-learn 1.8.0. Other supported versions may
produce small numerical differences. Further model choices after examining
test results require new final test data.

## Cancellation and local receiver

`Cancellation time` is editable from 1–120 seconds for the demonstration.
The initial 5-second value is only a UI default. It does not introduce a fixed
cancellation duration into your proposal. A server-side deadline controls the
outcome; changing the form after an alert starts affects the next alert only.

Each analysis run can create one alert. Replaying it does not create duplicates.
Analyze again to create a separate run. The browser polls the backend about every
0.8 seconds. The backend reconciles expired alerts when events are requested;
after restarting it reconciles stored deadlines on the next request. This local
prototype does not have an independent always-running notification dispatcher.

Events and predictions are saved in `runtime/events.sqlite3`, created on first
launch. The ZIP starts with an empty event history. The latest 100 events appear
in the UI; Export CSV includes all events. Cancelled events remain cancelled
after restart. To reset your demo history, stop the app and delete that database.

## Files and checks

| File/folder | Purpose |
|---|---|
| `app.py`, `core.py` | Local API, event storage, feature extraction and inference |
| `templates/`, `static/` | Frontend without external asset dependencies |
| `model/` | Trained tree and readable rules |
| `train_model.py` | Reproduce training, selection, baseline and held-out evaluation |
| `data/processed/` | All 66 Step 2 recordings, separated by participant split |
| `data/dataset_sources.json` | Original source URLs and source-file hashes |
| `data/processed_manifest.json` | Included processed-file hashes |
| `reports/` | Actual results, split, data quality and verification |
| `tests/test_demo.py` | API, validation, prediction and event-flow checks |
| `API.md`, `DEMO_SCRIPT.md` | Team integration and presentation guidance |

Run checks with:

```bat
..\.venv313\Scripts\python.exe -m unittest discover -s tests -v
```

## Dataset source

Martínez-Villaseñor et al. (2019), *UP-Fall Detection Dataset: A Multimodal
Approach*, Sensors 19(9), 1988. https://doi.org/10.3390/s19091988

Dataset site: https://sites.google.com/up.edu.mx/har-up/

The included CSVs are processed belt sensor excerpts from the publicly provided
dataset. Original synchronized raw files are not duplicated in this package;
`dataset_sources.json` records their source URLs and hashes. Those original
hashes differ from the processed-file hashes in `processed_manifest.json`.
