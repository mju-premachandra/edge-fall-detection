# Model evaluation

Model trained on participants 1–3; selected on 4; tested on 5–6.
Settings frozen before loading test data. Model SHA-256: `71d9f1cc35a9f9b2e5ae7e9c1345ead553b9c3e620d8c4891f82a3fe42f524c1`.

| Split | Windows | Model precision | Model recall | Model F1 | Baseline F1 |
|---|---:|---:|---:|---:|---:|
| train | 3237 | 0.579 | 0.947 | 0.719 | 0.284 |
| validation | 1147 | 0.278 | 1.000 | 0.435 | 0.172 |
| test | 2111 | 0.374 | 0.821 | 0.514 | 0.214 |

These are window metrics on overlapping windows. Event recall and scored non-fall exposure are reported separately in training_report.json.

## Fixed input protocol
{
  "window_seconds": 1.0,
  "step_seconds": 0.25,
  "min_samples": 10,
  "minimum_span_seconds": 0.8,
  "maximum_gap_seconds": 0.25,
  "target": "Whether the supplied activity tag at the window endpoint is 1-5 (fall); 6-11 are non-fall.",
  "unknown_labels": "Exclude any labelled training/evaluation window containing a tag outside 1-11.",
  "timing": "Use original elapsed timestamps. Do not resample, interpolate, or cross recording boundaries."
}

## Limits
- Six participants and one repetition per activity; only two held-out participants.
- Laboratory data; not validated for construction work or unsupervised use.
- Unknown labels and ineligible windows are excluded; irregular timing and gaps reduce coverage.
- A leaf score is not a calibrated real-world probability.
- Replay and manual synthetic input do not validate live sensors, physical haptics or on-device timing.
- Reusing held-out recordings for later model choices would turn them into development data; use new people for a new final test.