"""Third model: StandardScaler + RBF SVC, using the original participant split.

Select settings on subject 4 only. Subjects 5-6 are a previously inspected
comparison set, not a fresh independent test. No retuning after evaluation.
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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from core import FEATURES, POLICY, load_model, score_tree
from train_model import dataset, metrics, event_metrics
from svm_model import margins_svm, margin_to_score

ROOT = Path(__file__).resolve().parent
PLAN = {
    'algorithm': 'SVC', 'kernel': 'rbf', 'scaler': 'StandardScaler fit on training subjects only',
    'C_candidates': [0.1, 1.0, 10.0], 'gamma_candidates': ['scale', 0.01, 0.1],
    'threshold_candidates': [0.35, 0.5, 0.65], 'class_weight': 'balanced',
    'selection': 'Validation F2, then precision, smaller C, earlier gamma candidate, higher score threshold.',
    'split': {'train': [1, 2, 3], 'validation': [4], 'reused_test': [5, 6]},
    'score_transform': 'sigmoid(decision_function); display score only, NOT calibrated probability',
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
    xtrain = np.asarray(xtrain, dtype=np.float64)
    xval = np.asarray(xval, dtype=np.float64)
    scaler = StandardScaler()
    ztrain = scaler.fit_transform(xtrain)  # Learn mean/scale from training subjects ONLY.
    zval = scaler.transform(xval)          # Never fit scaling on validation or comparison data.
    candidates, best = [], None
    for c in PLAN['C_candidates']:
        for gamma_index, gamma in enumerate(PLAN['gamma_candidates']):
            clf = SVC(C=c, kernel='rbf', gamma=gamma, class_weight='balanced', cache_size=128)
            clf.fit(ztrain, ytrain)        # Actual supervised machine-learning training.
            if clf.fit_status_ != 0:
                raise RuntimeError('SVM solver did not converge.')
            scores = margin_to_score(clf.decision_function(zval))
            for threshold in PLAN['threshold_candidates']:
                measured = metrics(yval, (scores >= threshold).astype(int))
                candidates.append({'C': c, 'gamma': gamma, 'threshold': threshold, **measured})
                key = (measured['f2'], measured['precision'], -c, -gamma_index, threshold)
                if best is None or key > best[0]:
                    best = key, clf, threshold
            print(f'Fitted SVM: C={c}, gamma={gamma}, support vectors={len(clf.support_)}', flush=True)
    return scaler, best[1], best[2], candidates


def freeze_svm(scaler, clf, threshold, candidates, root=ROOT):
    root = Path(root)
    (root / 'model').mkdir(exist_ok=True)
    (root / 'reports').mkdir(exist_ok=True)
    assert list(clf.classes_) == [0, 1]
    model = {
        'model_id': 'fall-svm-v1', 'algorithm': 'SVC', 'kernel': 'rbf',
        'feature_names': FEATURES, 'window_policy': POLICY, 'decision_threshold': float(threshold),
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
        'support_vectors': clf.support_vectors_.tolist(), 'dual_coef': clf.dual_coef_[0].tolist(),
        'intercept': float(clf.intercept_[0]), 'gamma': float(clf._gamma),
        'score_note': 'Sigmoid of the SVM decision margin, for display only. NOT a calibrated fall probability; scores are not directly comparable across algorithms.',
        'training': {'settings': {'C': clf.C, 'kernel': clf.kernel, 'gamma': clf.gamma,
                                 'class_weight': clf.class_weight},
                     'support_vector_count': len(clf.support_), 'scaler_fit_windows': int(scaler.n_samples_seen_),
                     'subjects': [1, 2, 3], 'validation_subjects': [4], 'reused_test_subjects': [5, 6],
                     'sklearn': sklearn.__version__, 'numpy': np.__version__,
                     'python': platform.python_version(), 'plan': PLAN},
        'core_sha256': sha256(root / 'core.py'),
    }
    (root / 'model/trained_svm.json').write_text(json.dumps(model, indent=2, allow_nan=False), encoding='utf-8')
    (root / 'reports/svm_protocol.json').write_text(json.dumps(PLAN, indent=2), encoding='utf-8')
    pd.DataFrame(candidates).drop(columns=['confusion_matrix', 'note']).to_csv(
        root / 'reports/svm_validation_candidates.csv', index=False)
    print(f'Frozen selected SVM: C={clf.C}, gamma={clf.gamma}, threshold={threshold}', flush=True)
    return model


def extended_metrics(y, prediction):
    result = metrics(y, prediction)
    (tn, fp), (fn, tp) = result['confusion_matrix']
    result.update(accuracy=(tn + tp) / len(y), specificity=tn / (tn + fp) if tn + fp else 0.0)
    result['balanced_accuracy'] = (result['recall'] + result['specificity']) / 2
    return result


def evaluate_models(scaler, clf, model, training, validation, root=ROOT):
    root = Path(root)
    frozen_hash = sha256(root / 'model/trained_svm.json')
    comparison = dataset(root / 'data/processed/test')  # Only opened after selection/export.
    assert {r['subject'] for r in comparison[1]} == {5, 6}
    tree = load_model(root / 'model/trained_tree.json')
    forest = None
    if (root / 'model/trained_forest.json').exists():
        from random_forest import load_forest, score_forest
        forest = load_forest(root / 'model/trained_forest.json')
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'plan': PLAN,
              'feature_names': FEATURES, 'window_policy': POLICY, 'svm_sha256': frozen_hash,
              'original_tree_sha256': sha256(root / 'model/trained_tree.json'),
              'selected': {**model['training']['settings'], 'threshold': model['decision_threshold'],
                           'support_vectors': model['training']['support_vector_count']}, 'splits': {}}
    rows, max_error = [], 0.0
    for name, part in [('train', training), ('validation', validation), ('reused_test', comparison)]:
        frames, window_rows, x, y = part
        reference = clf.decision_function(scaler.transform(np.asarray(x, dtype=np.float64)))
        portable = margins_svm(model, x)
        np.testing.assert_allclose(portable, reference, rtol=0, atol=1e-9)
        max_error = max(max_error, float(np.max(np.abs(portable - reference))))
        methods = [('decision_tree', np.array([score_tree(tree, r) for r in x]), tree['decision_threshold']),
                   ('svm', margin_to_score(portable), model['decision_threshold'])]
        if forest is not None:
            methods.insert(1, ('random_forest', np.array([score_forest(forest, r) for r in x]), forest['decision_threshold']))
        answers = {}
        for method, scores, threshold in methods:
            prediction = (scores >= threshold).astype(int)
            measured, events = extended_metrics(y, prediction), event_metrics(frames, window_rows, prediction)
            answers[method] = {'window_metrics': measured, 'event_metrics': events}
            rows.append({'split': name, 'method': method, 'threshold': threshold,
                         **{k: measured[k] for k in ['windows', 'fall_windows', 'accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'f2']},
                         'detected_events': events['detected_fall_intervals'],
                         'labelled_fall_events': events['labelled_fall_intervals'],
                         'false_event_groups_in_nonfall_recordings': events['false_events_in_nonfall_recordings']})
        report['splits'][name] = answers
        print(f'Evaluated {name}: {len(y)} windows; portable SVM margins match sklearn.', flush=True)
    report['checks'] = {'portable_margins_match_sklearn': True, 'absolute_tolerance': 1e-9,
                        'maximum_absolute_margin_error': max_error,
                        'scaler_fit_windows': int(scaler.n_samples_seen_), 'scaler_fit_subjects': [1, 2, 3],
                        'activity_labels_used_as_features': False}
    report['limits'] = [PLAN['comparison_status'],
        'Only six participants and one trial per activity; overlapping windows are not independent events.',
        'Laboratory belt data; no live hardware or construction-worker validation.',
        'Unknown activity tags and unsuitable timing windows excluded by the original policy.',
        'SVM scores are sigmoid-transformed margins, not calibrated probabilities or comparable confidence across models.',
        'Accuracy alone can be misleading: only 56 of the 2,111 comparison windows are falls.',
        'No settings were changed after this comparison. New participants are needed for a fresh final test.',
        'Wearable memory, latency and energy use have not been measured.']
    (root / 'reports/svm_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    pd.DataFrame(rows).to_csv(root / 'reports/svm_comparison.csv', index=False)
    write_summary(report, root)
    assert sha256(root / 'model/trained_svm.json') == frozen_hash
    return report


def write_summary(report, root):
    lines = ['# SVM comparison', '', PLAN['comparison_status'], '',
             'Same UP-Fall belt subset, 12 features and causal one-second windows as the original models.',
             'StandardScaler was fit on subjects 1-3 only. Settings were selected on subject 4, then frozen.', '',
             '| Reused subjects 5-6 | Accuracy | Precision | Recall | F1 |', '|---|---:|---:|---:|---:|']
    for name, answer in report['splits']['reused_test'].items():
        m = answer['window_metrics']
        lines.append(f"| {name} | {m['accuracy']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.3f} |")
    lines += ['', 'Confusion matrices in svm_report.json use [[TN, FP], [FN, TP]].', '',
              '| Method | Detected fall intervals | False groups in non-fall recordings |', '|---|---:|---:|']
    for name, answer in report['splits']['reused_test'].items():
        e = answer['event_metrics']
        lines.append(f"| {name} | {e['detected_fall_intervals']}/{e['labelled_fall_intervals']} | {e['false_events_in_nonfall_recordings']} |")
    lines += ['', 'Event counts use the original one-to-one matching rule and short scored exposure, before cancellation.',
              'A high event recall can coexist with missed fall windows; they are different measurements.', '',
              '## Selected settings', '', '```json', json.dumps(report['selected'], indent=2), '```', '',
              '## Limitations', '', *['- ' + x for x in report['limits']], '',
              '## Source and documentation', '',
              '- UP-Fall dataset: https://doi.org/10.3390/s19091988',
              '- SVM documentation: https://scikit-learn.org/stable/modules/svm.html',
              '- StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html', '']
    (Path(root) / 'reports/SVM_COMPARISON.md').write_text('\n'.join(lines), encoding='utf-8')


def run_experiment(root=ROOT):
    root = Path(root)
    protected = ['core.py', 'train_model.py', 'model/trained_tree.json', 'reports/training_report.json',
                 'model/trained_forest.json', 'reports/random_forest_report.json']
    original = {n: sha256(root / n) for n in protected if (root / n).exists()}
    (root / 'reports').mkdir(exist_ok=True)
    (root / 'reports/svm_protocol.json').write_text(json.dumps(PLAN, indent=2), encoding='utf-8')
    training, validation = prepare_development(root)
    scaler, clf, threshold, candidates = fit_candidates(training, validation)
    model = freeze_svm(scaler, clf, threshold, candidates, root)
    report = evaluate_models(scaler, clf, model, training, validation, root)
    assert original == {n: sha256(root / n) for n in original}, 'An existing model or feature file changed.'
    print('\nSVM ready. Restart the application, then select Support Vector Machine (SVM).', flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    args = parser.parse_args()
    run_experiment(args.project)
