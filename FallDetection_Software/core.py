"""Shared causal sensor features and portable decision-tree inference."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ["acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_deg_s", "gyro_y_deg_s", "gyro_z_deg_s"]
FEATURES = ["acc_mean", "acc_std", "acc_min", "acc_max", "acc_range", "gyro_mean", "gyro_std", "gyro_max", "tilt_change", "acc_last", "gyro_last", "acc_diff_max"]
WINDOW_SECONDS = 1.0
STEP_SECONDS = 0.25
MIN_SAMPLES = 10
MIN_COVERAGE = 0.8
MAX_GAP = 0.25
POLICY = {"window_seconds": WINDOW_SECONDS, "step_seconds": STEP_SECONDS,
          "min_samples": MIN_SAMPLES, "minimum_span_seconds": MIN_COVERAGE,
          "maximum_gap_seconds": MAX_GAP,
          "target": "Whether the supplied activity tag at the window endpoint is 1-5 (fall); 6-11 are non-fall.",
          "unknown_labels": "Exclude any labelled training/evaluation window containing a tag outside 1-11.",
          "timing": "Use original elapsed timestamps. Do not resample, interpolate, or cross recording boundaries."}

class InputError(ValueError):
    pass

def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or len(frame) < MIN_SAMPLES:
        raise InputError(f"Supply at least {MIN_SAMPLES} timestamped readings; a single reading is insufficient.")
    if len(frame) > 20000:
        raise InputError("Use at most 20,000 readings per request.")
    frame = frame.copy()
    if "time_s" not in frame:
        if "timestamp" not in frame:
            raise InputError("CSV needs time_s (elapsed seconds), or an ISO timestamp column.")
        stamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        if stamps.isna().any():
            raise InputError("One or more timestamps are invalid.")
        frame["time_s"] = (stamps - stamps.iloc[0]).dt.total_seconds()
    missing = [name for name in ["time_s", *SENSORS] if name not in frame]
    if missing:
        raise InputError("Missing columns: " + ", ".join(missing) + ". Use the provided example CSV or a Step 2 belt CSV.")
    for name in ["time_s", *SENSORS]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    if not np.isfinite(frame[["time_s", *SENSORS]].to_numpy(float)).all():
        raise InputError("Readings must be finite numbers, with no missing values.")
    if not (np.diff(frame.time_s.to_numpy(float)) > 0).all():
        raise InputError("Timestamps must be strictly increasing; duplicates are not allowed.")
    frame["time_s"] -= float(frame.time_s.iloc[0])
    return frame

def feature_vector(window: pd.DataFrame) -> np.ndarray:
    values = window[SENSORS].to_numpy(dtype=float)
    acc = np.linalg.norm(values[:, :3], axis=1)
    gyro = np.linalg.norm(values[:, 3:], axis=1)
    first, last = values[0, :3], values[-1, :3]
    denom = np.linalg.norm(first) * np.linalg.norm(last)
    tilt = 0.0 if denom < 1e-9 else float(np.arccos(np.clip(np.dot(first, last) / denom, -1, 1)))
    return np.array([acc.mean(), acc.std(ddof=0), acc.min(), acc.max(), np.ptp(acc),
                     gyro.mean(), gyro.std(ddof=0), gyro.max(), tilt, acc[-1], gyro[-1],
                     np.abs(np.diff(acc)).max()], dtype=np.float32)

def windows(frame: pd.DataFrame, require_labels=False):
    """Yield valid windows ending at observed samples, using no future readings."""
    times = frame.time_s.to_numpy(float)
    next_end = float(times[0]) + WINDOW_SECONDS
    for end in range(len(frame)):
        t = float(times[end])
        if t + 1e-9 < next_end:
            continue
        next_end = t + STEP_SECONDS
        start = int(np.searchsorted(times, t-WINDOW_SECONDS, side="left"))
        win = frame.iloc[start:end+1]
        if len(win) < MIN_SAMPLES or t-float(times[start]) < MIN_COVERAGE:
            continue
        if np.diff(times[start:end+1]).max() > MAX_GAP:
            continue
        if not np.isfinite(win[SENSORS].to_numpy(float)).all():
            continue
        if require_labels and not win.frame_activity_id.isin(range(1, 12)).all():
            continue
        target = int(frame.frame_activity_id.iloc[end] in range(1, 6)) if require_labels else None
        yield {"start_s": float(times[start]), "end_s": t, "end_index": end,
               "features": feature_vector(win), "target": target, "samples": len(win)}

def score_tree(model, features):
    x = np.asarray(features, dtype=np.float32)
    tree = model["tree"]
    i = 0
    while tree["children_left"][i] != -1:
        i = tree["children_left"][i] if float(x[tree["feature"][i]]) <= tree["threshold"][i] else tree["children_right"][i]
    return float(tree["fall_score"][i])

def predict_windows(model, frame):
    answers = []
    for win in windows(frame):
        score = score_tree(model, win["features"])
        answers.append({"time_s": win["end_s"], "start_s": win["start_s"], "samples": win["samples"],
                        "fall_score": score, "suspected_fall": bool(score >= model["decision_threshold"]),
                        "features": dict(zip(FEATURES, map(float, win["features"])))})
    return answers

def load_model(path: Path):
    model = json.loads(path.read_text(encoding="utf-8"))
    if model["feature_names"] != FEATURES or model["window_policy"] != POLICY:
        raise RuntimeError("The saved model and feature policy do not match. Retrain using this version of core.py.")
    return model
