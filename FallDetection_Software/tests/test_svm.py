from pathlib import Path
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import create_app
from svm_model import load_svm, margin_to_score, predict_svm, margins_svm


class SvmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'events.sqlite3'
        self.client = create_app(self.db).test_client()
        self.model = load_svm(ROOT / 'model/trained_svm.json')
        self.csv = self.client.get('/api/examples/fall').json['csv']
        self.frame = pd.read_csv(io.StringIO(self.csv))

    def tearDown(self):
        self.temp.cleanup()

    def analyze(self, **body):
        response = self.client.post('/api/analyze', json={'model_id': 'fall-svm-v1', **body})
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        return response.json

    def event(self):
        answer = self.analyze(sample_id='Subject1Activity1Trial1_belt')
        positive = next(p for p in answer['predictions'] if p['suspected_fall'])
        response = self.client.post('/api/events', json={
            'decision_id': positive['decision_id'], 'model_id': 'fall-tree-v1', 'cancellation_seconds': 5})
        self.assertEqual(response.status_code, 201)
        return response.json['event']

    def test_catalog_and_report_are_for_svm(self):
        catalog = self.client.get('/api/models').json
        self.assertEqual({x['id'] for x in catalog['models']},
                         {'fall-tree-v1', 'fall-forest-v1', 'fall-svm-v1'})
        meta = self.client.get('/api/model?model_id=fall-svm-v1').json
        self.assertEqual(meta['algorithm'], 'SVC')
        self.assertIn('NOT a calibrated', meta['score_note'])
        self.assertIn('Exploratory', meta['evaluation_status'])
        report = json.loads((ROOT / 'reports/svm_report.json').read_text())
        self.assertEqual(meta['evaluation']['splits']['test']['model'],
                         report['splits']['reused_test']['svm']['window_metrics'])
        self.assertEqual(meta['training']['scaler_fit_windows'], 3237)

    def test_selected_svm_used_by_all_input_routes(self):
        direct = predict_svm(self.model, self.frame)
        csv = self.analyze(csv=self.csv)
        samples = self.analyze(samples=self.frame.to_dict('records'))
        uploaded = self.client.post('/api/analyze', data={
            'file': (io.BytesIO(self.csv.encode()), 'input.csv'), 'model_id': 'fall-svm-v1'},
            content_type='multipart/form-data')
        self.assertEqual(uploaded.status_code, 200)
        for answer in [csv, samples, uploaded.json]:
            self.assertEqual(answer['model_id'], 'fall-svm-v1')
            self.assertEqual([p['fall_score'] for p in answer['predictions']],
                             [p['fall_score'] for p in direct])
            self.assertTrue(all(p['model_id'] == 'fall-svm-v1' for p in answer['predictions']))
        last = self.client.post('/api/predict', json={
            'model_id': 'fall-svm-v1', 'samples': self.frame.to_dict('records')})
        self.assertEqual(last.status_code, 200)
        self.assertEqual(last.json['svm_margin'], direct[-1]['svm_margin'])
        self.assertEqual(last.json['model_id'], 'fall-svm-v1')
        self.analyze(sample_id='Subject1Activity6Trial1_belt')

    def test_labels_cannot_change_svm_prediction(self):
        normal = self.analyze(samples=self.frame.to_dict('records'))
        frame = self.frame.assign(frame_activity_id=20, subject_id=999, trial_activity_id=1)
        forged = self.analyze(samples=frame.to_dict('records'))
        self.assertEqual([p['fall_score'] for p in normal['predictions']],
                         [p['fall_score'] for p in forged['predictions']])

    def test_svm_alert_identity_survives_cancel_and_restart(self):
        event = self.event()
        self.assertEqual(event['model_id'], 'fall-svm-v1')
        self.client.get('/api/model?model_id=fall-tree-v1')
        cancelled = self.client.post('/api/events/' + event['id'] + '/cancel', json={})
        self.assertEqual(cancelled.json['event']['status'], 'cancelled')
        restarted = create_app(self.db).test_client()
        saved = restarted.get('/api/events').json['events'][0]
        self.assertEqual(saved['model_id'], 'fall-svm-v1')
        self.assertEqual(saved['status'], 'cancelled')

    def test_svm_expiry_and_acknowledgement(self):
        event = self.event()
        with patch('app.time.time', return_value=time.time() + 30):
            restarted = create_app(self.db).test_client()
            self.assertEqual(restarted.post('/api/events/' + event['id'] + '/cancel', json={}).status_code, 409)
            ack = restarted.post('/api/events/' + event['id'] + '/acknowledge', json={}).json['event']
            self.assertEqual(ack['model_id'], 'fall-svm-v1')
            self.assertEqual([h['status'] for h in ack['history']], ['pending', 'escalated', 'acknowledged'])
            self.assertIn('fall-svm-v1', restarted.get('/api/events/export.csv').get_data(as_text=True))

    def test_display_score_is_monotonic_and_numerically_stable(self):
        scores = margin_to_score(np.array([-1e6, -2, 0, 2, 1e6]))
        self.assertTrue(np.isfinite(scores).all())
        self.assertEqual(scores[2], 0.5)
        self.assertTrue((np.diff(scores) > 0).all())
        threshold = self.model['decision_threshold']
        margin_threshold = np.log(threshold / (1 - threshold))
        result = predict_svm(self.model, self.frame)
        for p in result:
            self.assertEqual(p['suspected_fall'], p['svm_margin'] >= margin_threshold)

    def test_invalid_model_scale_is_rejected(self):
        broken = dict(self.model)
        broken['scaler_scale'] = [0] * 12
        path = Path(self.temp.name) / 'broken.json'
        path.write_text(json.dumps(broken))
        with self.assertRaises(ValueError):
            load_svm(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
