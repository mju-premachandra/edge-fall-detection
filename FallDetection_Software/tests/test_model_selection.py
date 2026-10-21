from pathlib import Path
import io,json,shutil,sqlite3,sys,tempfile,time,unittest
from unittest.mock import patch
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import create_app
from core import load_model,predict_windows,validate_frame
from random_forest import load_forest,predict_forest

class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/'events.sqlite3'
        self.client=create_app(self.db).test_client()
        self.csv=(ROOT/'examples/fall_window.csv').read_text()
        self.frame=pd.read_csv(io.StringIO(self.csv))
    def tearDown(self):self.temp.cleanup()

    def analysis(self,model_id):
        response=self.client.post('/api/analyze',json={'csv':self.csv,'model_id':model_id})
        self.assertEqual(response.status_code,200,response.get_data(as_text=True))
        return response.json

    def test_catalog_and_metadata_identify_both_models(self):
        catalog=self.client.get('/api/models').json
        self.assertTrue({'fall-tree-v1','fall-forest-v1'}.issubset({m['id'] for m in catalog['models']}))
        self.assertEqual(catalog['default_model_id'],'fall-tree-v1')
        forest=self.client.get('/api/model?model_id=fall-forest-v1').json
        self.assertEqual(forest['algorithm'],'RandomForestClassifier')
        self.assertIn('Exploratory',forest['evaluation_status'])
        report=json.loads((ROOT/'reports/random_forest_report.json').read_text())
        self.assertEqual(forest['evaluation']['splits']['test']['model'],report['splits']['reused_test']['random_forest']['window_metrics'])

    def test_all_input_routes_use_actual_selected_model(self):
        for model_id in ['fall-tree-v1','fall-forest-v1']:
            model=load_forest() if model_id=='fall-forest-v1' else load_model(ROOT/'model/trained_tree.json')
            direct=(predict_forest(model,self.frame) if model_id=='fall-forest-v1' else predict_windows(model,validate_frame(self.frame)))
            pasted=self.analysis(model_id)
            samples=self.client.post('/api/analyze',json={'samples':self.frame.to_dict('records'),'model_id':model_id}).json
            upload=self.client.post('/api/analyze',data={'file':(io.BytesIO(self.csv.encode()),'input.csv'),'model_id':model_id},content_type='multipart/form-data').json
            for response in [pasted,samples,upload]:
                self.assertEqual(response['model_id'],model_id)
                self.assertEqual([p['fall_score'] for p in response['predictions']],[p['fall_score'] for p in direct])
                self.assertTrue(all(p['model_id']==model_id for p in response['predictions']))
            final=self.client.post('/api/predict',json={'samples':self.frame.to_dict('records'),'model_id':model_id})
            self.assertEqual(final.status_code,200)
            self.assertEqual(final.json['model_id'],model_id)
            self.assertEqual(final.json['fall_score'],direct[-1]['fall_score'])
            recorded=self.client.post('/api/analyze',json={'sample_id':'Subject1Activity1Trial1_belt','model_id':model_id})
            self.assertEqual(recorded.status_code,200)
            self.assertEqual(recorded.json['model_id'],model_id)

    def test_unknown_model_is_rejected(self):
        self.assertEqual(self.client.get('/api/model?model_id=not-a-model').status_code,400)
        self.assertEqual(self.client.post('/api/analyze',json={'csv':self.csv,'model_id':'fake'}).status_code,400)
        self.assertEqual(self.client.post('/api/analyze',json={'csv':self.csv,'model_id':{}}).status_code,400)

    def test_alert_model_identity_cancellation_timeout_and_restart(self):
        first=self.analysis('fall-forest-v1')
        positive=next(p for p in first['predictions'] if p['suspected_fall'])
        event=self.client.post('/api/events',json={'decision_id':positive['decision_id'],'model_id':'fall-tree-v1','cancellation_seconds':30}).json['event']
        self.assertEqual(event['model_id'],'fall-forest-v1')
        self.client.get('/api/model?model_id=fall-tree-v1')
        cancelled=self.client.post('/api/events/'+event['id']+'/cancel',json={})
        self.assertEqual(cancelled.json['event']['model_id'],'fall-forest-v1')
        self.assertEqual(cancelled.json['event']['status'],'cancelled')
        second=self.analysis('fall-forest-v1')
        p=next(p for p in second['predictions'] if p['suspected_fall'])
        e=self.client.post('/api/events',json={'decision_id':p['decision_id'],'cancellation_seconds':1}).json['event']
        with patch('app.time.time',return_value=time.time()+10):
            restarted=create_app(self.db).test_client()
            self.assertEqual(restarted.post('/api/events/'+e['id']+'/cancel',json={}).status_code,409)
            ack=restarted.post('/api/events/'+e['id']+'/acknowledge',json={}).json['event']
            self.assertEqual(ack['model_id'],'fall-forest-v1')
            self.assertEqual([t['status'] for t in ack['history']],['pending','escalated','acknowledged'])
            self.assertIn('fall-forest-v1',restarted.get('/api/events/export.csv').get_data(as_text=True))

    def test_legacy_database_is_migrated_without_deleting_history(self):
        legacy=Path(self.temp.name)/'legacy.sqlite3'
        with sqlite3.connect(legacy) as db:
            db.execute('CREATE TABLE decisions (id TEXT PRIMARY KEY,run_id TEXT,created REAL,fall INTEGER,score REAL,source TEXT,sensor_time REAL)')
            db.execute('INSERT INTO decisions VALUES (?,?,?,?,?,?,?)',('old-decision','old-run',time.time(),1,0.9,'old-recording',1.0))
        client=create_app(legacy).test_client()
        e=client.post('/api/events',json={'decision_id':'old-decision','cancellation_seconds':30})
        self.assertEqual(e.status_code,201)
        self.assertEqual(e.json['event']['model_id'],'fall-tree-v1')
        new=create_app(legacy).test_client()
        self.assertEqual(len(new.get('/api/events').json['events']),1)
        with sqlite3.connect(legacy) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM decisions').fetchone()[0],1)
            self.assertEqual(len(db.execute('PRAGMA table_info(decisions)').fetchall()),8)

    def test_original_app_can_still_start_without_forest_addon(self):
        minimal=Path(self.temp.name)/'minimal'
        for name in ['model/trained_tree.json','reports/training_report.json']:
            target=minimal/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/name,target)
        with patch('app.ROOT',minimal):
            client=create_app(Path(self.temp.name)/'minimal.sqlite3').test_client()
            self.assertEqual(len(client.get('/api/models').json['models']),1)
            self.assertEqual(client.post('/api/analyze',json={'csv':self.csv}).status_code,200)
            self.assertEqual(client.post('/api/analyze',json={'csv':self.csv,'model_id':'fall-forest-v1'}).status_code,400)

if __name__=='__main__':unittest.main(verbosity=2)
