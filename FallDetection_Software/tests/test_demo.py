from pathlib import Path
import io, json, sys, tempfile, time, unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import create_app
from core import windows, feature_vector, SENSORS, validate_frame

class DemoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/'events.sqlite3'
        self.app=create_app(self.db);self.app.config['TESTING']=True
        self.client=self.app.test_client()
    def tearDown(self):self.temp.cleanup()
    def analyze(self,sample='Subject1Activity1Trial1_belt'):
        response=self.client.post('/api/analyze',json={'sample_id':sample})
        self.assertEqual(response.status_code,200,response.get_data(as_text=True))
        return response.get_json()
    def event(self):
        analysis=self.analyze();positive=next(p for p in analysis['predictions'] if p['suspected_fall'])
        response=self.client.post('/api/events',json={'decision_id':positive['decision_id'],'cancellation_seconds':5,'worker_id':'TEST-001'})
        self.assertEqual(response.status_code,201,response.get_data(as_text=True))
        return positive,response.get_json()['event']

    def test_frontend_assets_health_and_actual_model(self):
        self.assertEqual(self.client.get('/').status_code,200)
        with self.client.get('/static/app.js') as response:
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.client.get('/api/health').json['status'],'ok')
        self.assertEqual(self.client.get('/api/model').json['training']['subjects'],[1,2,3])
        self.assertEqual(len(self.client.get('/api/samples').json['samples']),66)

    def test_real_sensor_input_generates_actual_predictions(self):
        result=self.analyze()
        self.assertTrue(any(p['suspected_fall'] for p in result['predictions']))
        self.assertEqual(result['input_rows'],195)
        self.assertEqual(len(result['predictions'][0]['features']),12)

    def test_csv_and_json_inputs_agree_and_labels_are_not_features(self):
        example=self.client.get('/api/examples/fall').json['csv']
        frame=pd.read_csv(io.StringIO(example))
        payload=frame.to_dict(orient='records')
        plain=self.client.post('/api/analyze',json={'samples':payload}).json
        for row in payload:row.update(frame_activity_id=7,trial_activity_id=6,subject_id=999)
        forged=self.client.post('/api/analyze',json={'samples':payload}).json
        uploaded=self.client.post('/api/analyze',data={'file':(io.BytesIO(example.encode()),'sensor.csv')},content_type='multipart/form-data').json
        a=[p['fall_score'] for p in plain['predictions']]
        self.assertEqual(a,[p['fall_score'] for p in forged['predictions']])
        self.assertEqual(a,[p['fall_score'] for p in uploaded['predictions']])
        self.assertTrue(any(p['suspected_fall'] for p in plain['predictions']))
        output=self.client.post('/api/predict',json={'samples':payload})
        self.assertEqual(output.status_code,200)
        self.assertEqual(output.json['fall_score'],a[-1])

    def test_invalid_inputs_return_actionable_errors(self):
        self.assertEqual(self.client.post('/api/predict',json={'samples':[{'acc_x_g':0}]}).status_code,400)
        self.assertEqual(self.client.post('/api/analyze',json={'sample_id':'../../app.py'}).status_code,400)
        rows=[{'time_s':i*.05,**dict(zip(SENSORS,[0,1,0,0,0,0]))} for i in range(25)]
        rows[5]['time_s']=rows[4]['time_s']
        self.assertIn('strictly increasing',self.client.post('/api/predict',json={'samples':rows}).json['error'])
        rows[5]['time_s']=.25;rows[5]['acc_x_g']='not a number'
        self.assertEqual(self.client.post('/api/predict',json={'samples':rows}).status_code,400)
        self.assertEqual(self.client.post('/api/analyze',json={'csv':'foo,bar\n1,2\n'}).status_code,400)

    def test_short_training_examples_demonstrate_both_workflows(self):
        walking=self.client.get('/api/examples/walking').json['csv']
        fall=self.client.get('/api/examples/fall').json['csv']
        normal=self.client.post('/api/analyze',json={'csv':walking}).json
        suspected=self.client.post('/api/analyze',json={'csv':fall}).json
        self.assertEqual(normal['suspected_fall_windows'],0)
        self.assertGreater(suspected['suspected_fall_windows'],0)

    def test_cancellation_is_durable_and_idempotent(self):
        decision,event=self.event()
        duplicate=self.client.post('/api/events',json={'decision_id':decision['decision_id'],'cancellation_seconds':5})
        self.assertTrue(duplicate.json['duplicate'])
        self.assertEqual(duplicate.json['event']['id'],event['id'])
        cancelled=self.client.post(f"/api/events/{event['id']}/cancel",json={})
        self.assertEqual(cancelled.json['event']['status'],'cancelled')
        self.assertTrue(self.client.post(f"/api/events/{event['id']}/cancel",json={}).json['duplicate'])
        with patch('app.time.time',return_value=time.time()+30):
            restarted=create_app(self.db).test_client()
            self.assertEqual(restarted.get('/api/events').json['events'][0]['status'],'cancelled')

    def test_expiry_escalation_late_cancel_acknowledgement_and_export(self):
        _,event=self.event();future=time.time()+30
        with patch('app.time.time',return_value=future):
            restarted=create_app(self.db).test_client()
            item=restarted.get('/api/events').json['events'][0]
            self.assertEqual(item['status'],'escalated')
            self.assertEqual(restarted.post(f"/api/events/{event['id']}/cancel",json={}).status_code,409)
            result=restarted.post(f"/api/events/{event['id']}/acknowledge",json={})
            self.assertEqual(result.json['event']['status'],'acknowledged')
            self.assertEqual([h['status'] for h in result.json['event']['history']],['pending','escalated','acknowledged'])
            self.assertTrue(restarted.post(f"/api/events/{event['id']}/acknowledge",json={}).json['duplicate'])
            self.assertIn('acknowledged',restarted.get('/api/events/export.csv').get_data(as_text=True))

    def test_fake_or_nonfall_prediction_cannot_create_alert(self):
        self.assertEqual(self.client.post('/api/events',json={'decision_id':'fake'}).status_code,400)
        result=self.analyze('Subject1Activity7Trial1_belt')
        safe=next(p for p in result['predictions'] if not p['suspected_fall'])
        self.assertEqual(self.client.post('/api/events',json={'decision_id':safe['decision_id']}).status_code,400)
        self.assertEqual(self.client.get('/api/events').json['events'],[])

    def test_causal_features_ignore_future_and_reject_unknown_training_labels(self):
        frame=pd.read_csv(ROOT/'data/processed/train/Subject1Activity1Trial1_belt.csv')
        before=list(windows(frame,require_labels=True))
        changed=frame.copy();boundary=before[0]['end_index']
        changed.loc[changed.index>boundary,SENSORS]=900
        after=list(windows(changed,require_labels=True))
        np.testing.assert_array_equal(before[0]['features'],after[0]['features'])
        frame['frame_activity_id']=20
        self.assertEqual(list(windows(frame,require_labels=True)),[])
        # Labels are not needed for inference.
        self.assertGreater(len(list(windows(frame.drop(columns='frame_activity_id')))),0)

if __name__=='__main__':unittest.main(verbosity=2)
