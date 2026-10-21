"""Portable Random Forest inference using the project's existing sensor features.

This adds a second model; app.py continues to use the original decision tree.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from core import FEATURES, POLICY, InputError, score_tree, validate_frame, windows

ROOT = Path(__file__).resolve().parent


def load_forest(path=None):
    path = Path(path) if path else ROOT / 'model/trained_forest.json'
    model = json.loads(path.read_text(encoding='utf-8'))
    if model.get('algorithm') != 'RandomForestClassifier':
        raise ValueError('This file is not an exported Random Forest.')
    if model['feature_names'] != FEATURES or model['window_policy'] != POLICY:
        raise ValueError('Forest feature policy differs from core.py. Use matching files.')
    if not model['trees']:
        raise ValueError('The forest has no trees.')
    return model


def score_forest(model, features):
    """Average the learned leaf scores, as sklearn RandomForestClassifier does."""
    return float(np.mean([score_tree({'tree': tree}, features) for tree in model['trees']]))


def predict_forest(model, frame):
    frame = validate_frame(frame)
    results = []
    for window in windows(frame):
        score = score_forest(model, window['features'])
        results.append({
            'time_s': window['end_s'], 'start_s': window['start_s'],
            'samples': window['samples'], 'fall_score': score,
            'suspected_fall': bool(score >= model['decision_threshold']),
            'features': dict(zip(FEATURES, map(float, window['features']))),
        })
    if not results:
        raise InputError('No eligible 1-second window. Use the same timing rules as the original app.')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict sensor CSV using the second ML model.')
    parser.add_argument('--csv', type=Path, required=True, help='Sensor CSV in the existing project schema.')
    parser.add_argument('--model', type=Path, default=ROOT / 'model/trained_forest.json')
    args = parser.parse_args()
    try:
        forest = load_forest(args.model)
        answers = predict_forest(forest, pd.read_csv(args.csv, nrows=20001))
    except (InputError, ValueError, OSError, pd.errors.ParserError) as error:
        parser.error(str(error))
    print(json.dumps({'model_id': forest['model_id'],
                      'decision_threshold': forest['decision_threshold'],
                      'predictions': answers}, indent=2, allow_nan=False))
