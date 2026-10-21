"""Portable binary RBF SVM inference; shared causal belt-sensor features.

The sigmoid-transformed margin is a display score, not a calibrated probability.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from core import FEATURES, POLICY, InputError, validate_frame, windows

ROOT = Path(__file__).resolve().parent


def load_svm(path=None):
    path = Path(path) if path else ROOT / 'model/trained_svm.json'
    model = json.loads(path.read_text(encoding='utf-8'))
    if model.get('algorithm') != 'SVC' or model.get('kernel') != 'rbf':
        raise ValueError('Expected an exported binary RBF SVM.')
    if model['feature_names'] != FEATURES or model['window_policy'] != POLICY:
        raise ValueError('SVM feature policy differs from core.py. Use matching files.')
    support, dual, mean, scale = _arrays(model)
    if support.ndim != 2 or support.shape[1] != len(FEATURES) or not len(support):
        raise ValueError('Invalid SVM support-vector dimensions.')
    if dual.shape != (len(support),) or mean.shape != (len(FEATURES),) or scale.shape != mean.shape:
        raise ValueError('Invalid SVM scaling or coefficient dimensions.')
    if not all(np.isfinite(v).all() for v in [support, dual, mean, scale]) or np.any(scale <= 0):
        raise ValueError('Invalid SVM coefficients or scaling.')
    if not np.isfinite(model['gamma']) or model['gamma'] <= 0:
        raise ValueError('Invalid RBF gamma.')
    if not np.isfinite(model['intercept']) or not 0 < model['decision_threshold'] < 1:
        raise ValueError('Invalid SVM intercept or threshold.')
    return model


def _arrays(model):
    return tuple(np.asarray(model[key], dtype=np.float64) for key in
                 ['support_vectors', 'dual_coef', 'scaler_mean', 'scaler_scale'])


def margin_to_score(margin):
    """Stable sigmoid for display only: no probability calibration is performed."""
    values = np.asarray(margin, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


def margins_svm(model, features):
    support, dual, mean, scale = _arrays(model)
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
        raise InputError('SVM needs finite rows of the 12 shared sensor features.')
    z = (x - mean) / scale
    result = np.empty(len(z), dtype=np.float64)
    for start in range(0, len(z), 64):
        stop = min(start + 64, len(z))
        distance = np.sum((z[start:stop, None, :] - support[None, :, :]) ** 2, axis=2)
        result[start:stop] = np.exp(-model['gamma'] * distance) @ dual + model['intercept']
    return result


def score_svm(model, features):
    return float(margin_to_score(margins_svm(model, np.asarray(features).reshape(1, -1)))[0])


def predict_svm(model, frame):
    frame = validate_frame(frame)
    eligible = list(windows(frame))
    if not eligible:
        raise InputError('No eligible 1-second window. Use the original timing rules.')
    margins = margins_svm(model, np.stack([w['features'] for w in eligible]))
    scores = margin_to_score(margins)
    return [{'time_s': w['end_s'], 'start_s': w['start_s'], 'samples': w['samples'],
             'fall_score': float(score), 'svm_margin': float(margin),
             'suspected_fall': bool(score >= model['decision_threshold']),
             'features': dict(zip(FEATURES, map(float, w['features'])))}
            for w, score, margin in zip(eligible, scores, margins)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict a sensor CSV with the trained SVM.')
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=ROOT / 'model/trained_svm.json')
    args = parser.parse_args()
    try:
        model = load_svm(args.model)
        answers = predict_svm(model, pd.read_csv(args.csv, nrows=20001))
    except (InputError, ValueError, OSError, pd.errors.ParserError) as error:
        parser.error(str(error))
    print(json.dumps({'model_id': model['model_id'], 'score_note': model['score_note'],
                      'decision_threshold': model['decision_threshold'],
                      'predictions': answers}, indent=2, allow_nan=False))
