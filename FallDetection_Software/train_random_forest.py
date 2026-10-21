"""Second-model experiment: keep the existing app and decision tree intact.

Same features, windows and participant split. Settings use subject 4 only.
Subjects 5-6 have already been inspected, so this is a reused-set comparison.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from core import FEATURES, POLICY, load_model, score_tree
from train_model import dataset, metrics, event_metrics
from random_forest import score_forest

ROOT = Path(__file__).resolve().parent
PLAN = {
    'algorithm': 'RandomForestClassifier',
    'n_estimators': 100,
    'max_depth_candidates': [5, 8, None],
    'min_samples_leaf_candidates': [2, 5],
    'threshold_candidates': [0.35, 0.5, 0.65],
    'max_features': 'sqrt',
    'class_weight': 'balanced_subsample',
    'random_state': 42,
    'selection': 'Validation F2, then precision, shallower depth, larger minimum leaf, higher threshold.',
    'split': {'train': [1, 2, 3], 'validation': [4], 'reused_test': [5, 6]},
    'comparison_status': 'Exploratory comparison on previously inspected subjects 5-6. Not a new independent final test.',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_development(root=ROOT):
    root = Path(root)
    training = dataset(root / 'data/processed/train')
    validation = dataset(root / 'data/processed/validation')
    assert {r['subject'] for r in training[1]} == {1, 2, 3}
    assert {r['subject'] for r in validation[1]} == {4}
    print(f'Training: {len(training[3])} windows; validation: {len(validation[3])} windows.', flush=True)
    return training, validation


def fit_candidates(training, validation):
    xtrain, ytrain = training[2:]
    xval, yval = validation[2:]
    candidates = []
    best = None
    for depth in PLAN['max_depth_candidates']:
        for leaf in PLAN['min_samples_leaf_candidates']:
            clf = RandomForestClassifier(
                n_estimators=PLAN['n_estimators'], max_depth=depth,
                min_samples_leaf=leaf, max_features=PLAN['max_features'],
                class_weight=PLAN['class_weight'], random_state=PLAN['random_state'],
                n_jobs=1,
            )
            clf.fit(xtrain, ytrain)  # The actual ML training step.
            scores = clf.predict_proba(xval)[:, list(clf.classes_).index(1)]
            for threshold in PLAN['threshold_candidates']:
                measured = metrics(yval, (scores >= threshold).astype(int))
                candidates.append({'max_depth': depth, 'min_samples_leaf': leaf,
                                   'threshold': threshold, **measured})
                key = (measured['f2'], measured['precision'],
                       -(depth if depth is not None else 10**6), leaf, threshold)
                if best is None or key > best[0]:
                    best = (key, clf, threshold)
            print(f'Fitted 100 trees: depth={depth}, minimum leaf={leaf}', flush=True)
    return best[1], best[2], candidates


def freeze_forest(clf, threshold, candidates, root=ROOT):
    """Write the selected model before reading comparison-set sensor values."""
    root = Path(root)
    (root / 'model').mkdir(exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    trees = []
    fall_index = list(clf.classes_).index(1)
    for estimator in clf.estimators_:
        tree = estimator.tree_
        value = tree.value[:, 0, :]
        scores = value[:, fall_index] / value.sum(axis=1)
        trees.append({'children_left': tree.children_left.tolist(),
                      'children_right': tree.children_right.tolist(),
                      'feature': tree.feature.tolist(), 'threshold': tree.threshold.tolist(),
                      'fall_score': scores.tolist()})
    model = {
        'model_id': 'fall-forest-v1', 'algorithm': 'RandomForestClassifier',
        'feature_names': FEATURES, 'window_policy': POLICY,
        'decision_threshold': float(threshold), 'trees': trees,
        'score_note': 'Mean weighted leaf class fractions; not a calibrated real-world fall probability.',
        'training': {'settings': clf.get_params(), 'plan': PLAN,
                     'sklearn': sklearn.__version__, 'numpy': np.__version__,
                     'python': platform.python_version()},
        'core_sha256': sha256(root / 'core.py'),
    }
    (root / 'model/trained_forest.json').write_text(json.dumps(model, indent=2, allow_nan=False), encoding='utf-8')
    (root / 'reports/random_forest_protocol.json').write_text(json.dumps(PLAN, indent=2), encoding='utf-8')
    # Store scalar columns in a compact, readable candidate table.
    pd.DataFrame(candidates).drop(columns=['confusion_matrix', 'note']).to_csv(
        root / 'reports/random_forest_validation_candidates.csv', index=False)
    pd.DataFrame({'feature': FEATURES, 'training_impurity_importance': clf.feature_importances_}).sort_values(
        'training_impurity_importance', ascending=False).to_csv(root / 'reports/random_forest_feature_importance.csv', index=False)
    print(f'Selected settings frozen: depth={clf.max_depth}, leaf={clf.min_samples_leaf}, threshold={threshold}', flush=True)
    return model


def extended_metrics(y, prediction):
    result = metrics(y, prediction)
    (tn, fp), (fn, tp) = result['confusion_matrix']
    result['accuracy'] = (tn + tp) / len(y)
    result['specificity'] = tn / (tn + fp) if tn + fp else 0.0
    result['balanced_accuracy'] = (result['recall'] + result['specificity']) / 2
    return result


def evaluate_models(clf, forest, training, validation, root=ROOT):
    root = Path(root)
    comparison = dataset(root / 'data/processed/test')
    assert {r['subject'] for r in comparison[1]} == {5, 6}
    tree = load_model(root / 'model/trained_tree.json')
    tree_hash = sha256(root / 'model/trained_tree.json')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'plan': PLAN,
              'feature_names': FEATURES, 'window_policy': POLICY,
              'forest_sha256': sha256(root / 'model/trained_forest.json'),
              'original_tree_sha256': tree_hash,
              'selected': {'n_estimators': len(forest['trees']), 'max_depth': clf.max_depth,
                           'min_samples_leaf': clf.min_samples_leaf,
                           'threshold': forest['decision_threshold']}, 'splits': {}}
    rows = []
    for name, part in [('train', training), ('validation', validation), ('reused_test', comparison)]:
        frames, windows_rows, x, y = part
        sklearn_scores = clf.predict_proba(x)[:, list(clf.classes_).index(1)]
        portable_scores = np.array([score_forest(forest, row) for row in x])
        np.testing.assert_allclose(portable_scores, sklearn_scores, rtol=0, atol=1e-12)
        tree_scores = np.array([score_tree(tree, row) for row in x])
        answers = {}
        for method, scores, threshold in [
            ('decision_tree', tree_scores, tree['decision_threshold']),
            ('random_forest', portable_scores, forest['decision_threshold'])
        ]:
            predicted = (scores >= threshold).astype(int)
            measured = extended_metrics(y, predicted)
            events = event_metrics(frames, windows_rows, predicted)
            answers[method] = {'window_metrics': measured, 'event_metrics': events}
            rows.append({'split': name, 'method': method, 'threshold': threshold,
                         **{k: measured[k] for k in ['windows', 'fall_windows', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'f2']},
                         'detected_events': events['detected_fall_intervals'],
                         'labelled_fall_events': events['labelled_fall_intervals'],
                         'false_event_groups_in_nonfall_recordings': events['false_events_in_nonfall_recordings']})
        report['splits'][name] = answers
        print(f'Evaluated {name}: {len(y)} windows; portable scores match sklearn.', flush=True)
    report['checks'] = {'portable_predictions_match_sklearn': True, 'tolerance': 1e-12,
                        'participant_split': PLAN['split'], 'activity_labels_used_as_features': False}
    report['limits'] = [
        PLAN['comparison_status'],
        'Only six participants, one trial per activity, and overlapping windows; not independent events.',
        'Laboratory belt data, not construction-worker or live-hardware validation.',
        'Unknown labels and unsuitable timing windows excluded by the original policy.',
        'A forest score is not a calibrated probability; accuracy alone is misleading on this imbalanced subset.',
        'No architecture or threshold was adjusted after this comparison evaluation.',
        'Model size and actual wearable deployment need separate assessment.',
    ]
    (root / 'reports/random_forest_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(rows).to_csv(root / 'reports/random_forest_comparison.csv', index=False)
    write_summary(report, root)
    assert sha256(root / 'model/trained_tree.json') == tree_hash
    return report


def write_summary(report, root):
    lines = ['# Random Forest comparison', '', report['plan']['comparison_status'], '',
             'Same six-axis belt data, 12 features and causal windows as the original model.',
             'Training subjects 1-3; settings selected using subject 4 only.', '',
             '| Reused subjects 5-6 | Accuracy | Precision | Recall | F1 |',
             '|---|---:|---:|---:|---:|']
    for name, result in report['splits']['reused_test'].items():
        m = result['window_metrics']
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.3f} |")
    lines += ['', '## Event comparison (before cancellation)', '',
              '| Method | Detected tagged fall intervals | False groups in non-fall recordings |',
              '|---|---:|---:|']
    for name, result in report['splits']['reused_test'].items():
        e = result['event_metrics']
        lines.append(f"| {name} | {e['detected_fall_intervals']}/{e['labelled_fall_intervals']} | {e['false_events_in_nonfall_recordings']} |")
    lines += ['', 'These event counts use the original matching rule and short scored exposure; they do not establish an operational false-alarm rate.',
              '', '## Selected forest', '', json.dumps(report['selected'], indent=2), '',
              '## Limits', '', *['- ' + limit for limit in report['limits']], '',
              'The current Flask frontend still uses the original decision tree. Use random_forest.py or the notebook to try the second model.']
    (root / 'reports/RANDOM_FOREST_COMPARISON.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_experiment(root=ROOT):
    root = Path(root)
    protected = ['app.py', 'core.py', 'train_model.py', 'model/trained_tree.json', 'reports/training_report.json']
    original = {name: sha256(root / name) for name in protected}
    (root / 'reports').mkdir(exist_ok=True)
    # Write the candidate plan before fitting any model or opening comparison data.
    (root / 'reports/random_forest_protocol.json').write_text(json.dumps(PLAN, indent=2), encoding='utf-8')
    training, validation = prepare_development(root)
    clf, threshold, candidates = fit_candidates(training, validation)
    forest = freeze_forest(clf, threshold, candidates, root)
    report = evaluate_models(clf, forest, training, validation, root)
    assert original == {name: sha256(root / name) for name in protected}, 'An original project file changed.'
    print('\nOriginal model and application preserved. Open reports/RANDOM_FOREST_COMPARISON.md.', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    run_experiment(args.project)
