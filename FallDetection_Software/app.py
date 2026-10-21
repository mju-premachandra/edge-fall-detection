"""Local frontend + sensor prediction API + durable software alert workflow."""
from __future__ import annotations
from pathlib import Path
import argparse, csv, io, json, os, sqlite3, threading, time, uuid, webbrowser
from datetime import datetime, timezone
from contextlib import contextmanager
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, Response
from werkzeug.exceptions import HTTPException
from core import SENSORS, POLICY, InputError, validate_frame, predict_windows, load_model

ROOT=Path(__file__).resolve().parent

def create_app(db_path=None):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=5*1024*1024
    app.config['DATABASE']=str(db_path or ROOT/'runtime/events.sqlite3')
    Path(app.config['DATABASE']).parent.mkdir(parents=True,exist_ok=True)
    model=load_model(ROOT/'model/trained_tree.json')
    report=json.loads((ROOT/'reports/training_report.json').read_text())
    models={model['model_id']:{'model':model,'predict':predict_windows,'label':'Decision Tree',
                             'report':report,'comparison_label':'Acceleration threshold',
                             'evaluation_status':'Original held-out evaluation on subjects 5–6.'}}
    model_messages=[]
    forest_path=ROOT/'model/trained_forest.json'
    if forest_path.exists():
        try:
            from random_forest import load_forest, predict_forest
            forest=load_forest(forest_path)
            forest_report=json.loads((ROOT/'reports/random_forest_report.json').read_text(encoding='utf-8'))
            # Adapt the saved comparison report to the existing UI evaluation schema.
            forest_evaluation={'splits':{},'limits':forest_report['limits']}
            for source,target in [('train','train'),('validation','validation'),('reused_test','test')]:
                measured=forest_report['splits'][source]
                forest_evaluation['splits'][target]={
                    'model':measured['random_forest']['window_metrics'],
                    'baseline':measured['decision_tree']['window_metrics'],
                    'model_events':measured['random_forest']['event_metrics']}
            models[forest['model_id']]={'model':forest,'predict':predict_forest,'label':'Random Forest',
                'report':forest_evaluation,'comparison_label':'Decision Tree',
                'evaluation_status':forest_report['plan']['comparison_status']}
        except (ImportError,OSError,ValueError,KeyError) as error:
            model_messages.append('Random Forest could not load. Check that the full add-on and its report are installed: '+str(error))
    else:
        model_messages.append('Install the Random Forest add-on to enable the second model.')

    svm_path=ROOT/'model/trained_svm.json'
    if svm_path.exists():
        try:
            from svm_model import load_svm, predict_svm
            svm=load_svm(svm_path)
            svm_report=json.loads((ROOT/'reports/svm_report.json').read_text(encoding='utf-8'))
            svm_evaluation={'splits':{},'limits':svm_report['limits']}
            for source,target in [('train','train'),('validation','validation'),('reused_test','test')]:
                measured=svm_report['splits'][source]
                svm_evaluation['splits'][target]={
                    'model':measured['svm']['window_metrics'],
                    'baseline':measured['decision_tree']['window_metrics'],
                    'model_events':measured['svm']['event_metrics']}
            models[svm['model_id']]={'model':svm,'predict':predict_svm,'label':'Support Vector Machine (SVM)',
                'report':svm_evaluation,'comparison_label':'Decision Tree',
                'evaluation_status':svm_report['plan']['comparison_status']}
        except (ImportError,OSError,ValueError,KeyError) as error:
            model_messages.append('SVM could not load. Check the SVM model, inference code and report: '+str(error))

    def selected_model(model_id=None):
        model_id=model['model_id'] if model_id is None else model_id
        if not isinstance(model_id,str) or model_id not in models:
            raise InputError('Choose an available model: '+', '.join(models))
        return models[model_id]

    @contextmanager
    def connect():
        db=sqlite3.connect(app.config['DATABASE'],timeout=10)
        db.row_factory=sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    with connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created REAL NOT NULL, fall INTEGER NOT NULL, score REAL NOT NULL, source TEXT NOT NULL, sensor_time REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, decision_id TEXT NOT NULL, worker_id TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL, deadline REAL NOT NULL, score REAL NOT NULL, source TEXT NOT NULL, model_id TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS transitions (id INTEGER PRIMARY KEY AUTOINCREMENT,event_id TEXT NOT NULL,status TEXT NOT NULL,at REAL NOT NULL);
        ''')
        # Upgrade existing local histories without deleting their decisions/events.
        columns={row['name'] for row in db.execute('PRAGMA table_info(decisions)')}
        if 'model_id' not in columns:
            db.execute("ALTER TABLE decisions ADD COLUMN model_id TEXT NOT NULL DEFAULT 'fall-tree-v1'")

    def refresh(db):
        now=time.time()
        expired=db.execute("SELECT id FROM events WHERE status='pending' AND deadline<=?",(now,)).fetchall()
        for row in expired:
            changed=db.execute("UPDATE events SET status='escalated' WHERE id=? AND status='pending'",(row['id'],))
            if changed.rowcount:
                db.execute("INSERT INTO transitions(event_id,status,at) VALUES (?,?,?)",(row['id'],'escalated',now))

    def event_json(db,row):
        value=dict(row)
        value['remaining_seconds']=max(0.0,row['deadline']-time.time()) if row['status']=='pending' else 0.0
        value['created_iso']=datetime.fromtimestamp(row['created'],timezone.utc).isoformat()
        value['history']=[dict(x) for x in db.execute('SELECT status,at FROM transitions WHERE event_id=? ORDER BY id',(row['id'],)).fetchall()]
        value['haptic_mode']='software visual prompt; no physical actuator'
        value['delivery_mode']='local backend receiver; no external notification sent'
        return value

    def json_body():
        body=request.get_json(silent=True)
        if not isinstance(body,dict):raise InputError('Send a JSON object.')
        return body

    def input_frame(body):
        rows=body.get('samples')
        if not isinstance(rows,list) or not all(isinstance(row,dict) for row in rows):
            raise InputError('samples must be a list of timestamped sensor objects.')
        return validate_frame(pd.DataFrame(rows))

    def analyzed(frame,source,model_id=None):
        chosen=selected_model(model_id)
        active_model=chosen['model']
        frame=validate_frame(frame)
        predictions=chosen['predict'](active_model,frame)
        if not predictions:
            raise InputError('No valid 1-second window. Supply readings covering at least 1 second, with at least 10 readings per window and no gaps over 0.25 s.')
        run_id=str(uuid.uuid4());now=time.time()
        with connect() as db:
            for p in predictions:
                p['decision_id']=str(uuid.uuid4())
                p['run_id']=run_id
                p['model_id']=active_model['model_id']
                db.execute('INSERT INTO decisions (id,run_id,created,fall,score,source,sensor_time,model_id) VALUES (?,?,?,?,?,?,?,?)',
                           (p['decision_id'],run_id,now,int(p['suspected_fall']),p['fall_score'],source,p['time_s'],active_model['model_id']))
        # Return sensor-only chart data. Labels/IDs never enter the model.
        values=frame[SENSORS].to_numpy(float)
        trace=[{'time_s':float(t),'magnitude_g':float(a),'gyro_deg_s':float(g)} for t,a,g in
               zip(frame.time_s,np.linalg.norm(values[:,:3],axis=1),np.linalg.norm(values[:,3:],axis=1))]
        return {'run_id':run_id,'model_id':active_model['model_id'],'source':source,'input_rows':len(frame),
                'duration_s':float(frame.time_s.iloc[-1]),'decision_threshold':active_model['decision_threshold'],
                'predictions':predictions,'trace':trace,'suspected_fall_windows':sum(p['suspected_fall'] for p in predictions),
                'note':'Predictions use only sensor values in the current and preceding 1 second. No activity label is used.'}

    @app.errorhandler(InputError)
    def bad_input(error):return jsonify(error=str(error)),400

    @app.errorhandler(HTTPException)
    def http_error(error):return jsonify(error=error.description),error.code

    @app.after_request
    def local_headers(response):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
        return response

    @app.get('/')
    def index():return render_template('index.html')

    @app.get('/api/health')
    def health():return jsonify(status='ok',model_id=model['model_id'],available_models=list(models),backend='Flask',mode='software demonstration')

    @app.get('/api/models')
    def available_models():
        return jsonify(default_model_id=model['model_id'],messages=model_messages,
                       models=[{'id':key,'label':entry['label'],'algorithm':entry['model']['algorithm'],
                                'threshold':entry['model']['decision_threshold']} for key,entry in models.items()])

    @app.get('/api/model')
    def metadata():
        chosen=selected_model(request.args.get('model_id'))
        active_model=chosen['model']
        training=dict(active_model['training'])
        training.setdefault('subjects',[1,2,3])
        return jsonify(model_id=active_model['model_id'],label=chosen['label'],algorithm=active_model['algorithm'],
                       threshold=active_model['decision_threshold'],window_policy=POLICY,
                       feature_names=active_model['feature_names'],score_note=active_model['score_note'],
                       training=training,evaluation=chosen['report'],comparison_label=chosen['comparison_label'],
                       evaluation_status=chosen['evaluation_status'])

    @app.get('/api/samples')
    def samples():
        entries=[]
        names={1:'Forward fall · hands',2:'Forward fall · knees',3:'Backward fall',4:'Sideways fall',5:'Fall while sitting',
               6:'Walking',7:'Standing',8:'Sitting',9:'Picking up an object',10:'Jumping',11:'Lying'}
        for path in sorted((ROOT/'data/processed').glob('*/*_belt.csv')):
            import re
            match=re.fullmatch(r'Subject(\d+)Activity(\d+)Trial1_belt',path.stem)
            if match:
                subject,activity=map(int,match.groups())
                entries.append({'id':path.stem,'subject':subject,'activity':activity,'split':path.parent.name,
                                'label':f'{names[activity]} — subject {subject} ({path.parent.name})'})
        entries.sort(key=lambda x:(x['subject'],x['activity']))
        return jsonify(samples=entries)

    def sample_path(sample_id):
        if not isinstance(sample_id,str) or len(sample_id)>100:raise InputError('Choose a valid recording.')
        # Resolve only known file stems; input is never treated as a path.
        for path in (ROOT/'data/processed').glob('*/*_belt.csv'):
            if path.stem==sample_id:return path
        raise InputError('Recording not found.')

    @app.get('/api/samples/<sample_id>/csv')
    def sample_csv(sample_id):
        frame=pd.read_csv(sample_path(sample_id))
        return Response(frame[['time_s',*SENSORS]].to_csv(index=False),mimetype='text/csv',
                        headers={'Content-Disposition':f'attachment; filename="{sample_id}_input.csv"'})

    @app.get('/api/examples/<kind>')
    def example(kind):
        if kind not in {'fall','walking'}:raise InputError('Choose fall or walking.')
        filename='Subject1Activity1Trial1_belt' if kind=='fall' else 'Subject1Activity6Trial1_belt'
        frame=pd.read_csv(sample_path(filename))
        # Representative known training windows, selected for demonstrations only.
        end=4.7 if kind=='fall' else 4.5
        window=frame[(frame.time_s>=end-1.5)&(frame.time_s<=end)].copy()
        window['time_s']-=float(window.time_s.iloc[0])
        return jsonify(csv=window[['time_s',*SENSORS]].to_csv(index=False),source='UP-Fall subject 1 training example',kind=kind)

    @app.post('/api/analyze')
    def analyze():
        if request.files:
            model_id=request.form.get('model_id')
            file=request.files.get('file')
            if file is None:raise InputError('Choose a CSV file.')
            try:frame=pd.read_csv(file,nrows=20001)
            except Exception as exc:raise InputError('Could not read CSV. Use a Step 2 belt CSV or the downloaded example.') from exc
            source='uploaded CSV'
        else:
            body=json_body()
            model_id=body.get('model_id')
            if 'sample_id' in body:
                path=sample_path(body['sample_id']);frame=pd.read_csv(path);source=path.stem
            elif 'csv' in body:
                if not isinstance(body['csv'],str) or len(body['csv'])>2_000_000:raise InputError('Paste CSV text under 2 MB.')
                try:frame=pd.read_csv(io.StringIO(body['csv']),nrows=20001)
                except Exception as exc:raise InputError('Could not read the pasted CSV.') from exc
                source='manual CSV input'
            else:frame=input_frame(body);source='manual sensor samples'
        return jsonify(analyzed(frame,source,model_id))

    @app.post('/api/predict')
    def predict():
        body=json_body()
        result=analyzed(input_frame(body),'API sensor input',body.get('model_id'))
        return jsonify(**result['predictions'][-1],
                       label='Suspected fall' if result['predictions'][-1]['suspected_fall'] else 'No fall predicted')

    @app.post('/api/events')
    def create_event():
        body=json_body();decision_id=body.get('decision_id')
        if not isinstance(decision_id,str):raise InputError('An actual backend prediction decision_id is required.')
        try:duration=float(body.get('cancellation_seconds',5))
        except (TypeError,ValueError):raise InputError('Cancellation time must be numeric.')
        if not np.isfinite(duration) or not 1<=duration<=120:raise InputError('Cancellation time must be between 1 and 120 seconds.')
        worker=body.get('worker_id','DEMO-001')
        if not isinstance(worker,str) or not 1<=len(worker.strip())<=64:raise InputError('Worker ID must contain 1-64 characters.')
        with connect() as db:
            db.execute('BEGIN IMMEDIATE');refresh(db)
            decision=db.execute('SELECT * FROM decisions WHERE id=?',(decision_id,)).fetchone()
            if not decision or not decision['fall']:raise InputError('Only a suspected-fall prediction can open a cancellation window.')
            previous=db.execute('SELECT * FROM events WHERE run_id=?',(decision['run_id'],)).fetchone()
            if previous:return jsonify(event=event_json(db,previous),duplicate=True)
            now=time.time();event_id=str(uuid.uuid4())
            db.execute('INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (event_id,decision['run_id'],decision_id,worker.strip(),'pending',now,now+duration,decision['score'],decision['source'],decision['model_id']))
            db.execute('INSERT INTO transitions(event_id,status,at) VALUES (?,?,?)',(event_id,'pending',now))
            row=db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()
            return jsonify(event=event_json(db,row),duplicate=False),201

    @app.get('/api/events')
    def events():
        with connect() as db:
            refresh(db)
            rows=db.execute('SELECT * FROM events ORDER BY created DESC LIMIT 100').fetchall()
            return jsonify(events=[event_json(db,row) for row in rows])

    @app.post('/api/events/<event_id>/cancel')
    def cancel(event_id):
        with connect() as db:
            db.execute('BEGIN IMMEDIATE');refresh(db)
            row=db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()
            if not row:return jsonify(error='Event not found.'),404
            if row['status']=='cancelled':return jsonify(event=event_json(db,row),duplicate=True)
            if row['status']!='pending':return jsonify(error='The cancellation window has closed.',event=event_json(db,row)),409
            db.execute("UPDATE events SET status='cancelled' WHERE id=?",(event_id,))
            db.execute('INSERT INTO transitions(event_id,status,at) VALUES (?,?,?)',(event_id,'cancelled',time.time()))
            return jsonify(event=event_json(db,db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()))

    @app.post('/api/events/<event_id>/acknowledge')
    def acknowledge(event_id):
        with connect() as db:
            db.execute('BEGIN IMMEDIATE');refresh(db)
            row=db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()
            if not row:return jsonify(error='Event not found.'),404
            if row['status']=='acknowledged':return jsonify(event=event_json(db,row),duplicate=True)
            if row['status']!='escalated':return jsonify(error='Only escalated alerts can be acknowledged.'),409
            db.execute("UPDATE events SET status='acknowledged' WHERE id=?",(event_id,))
            db.execute('INSERT INTO transitions(event_id,status,at) VALUES (?,?,?)',(event_id,'acknowledged',time.time()))
            return jsonify(event=event_json(db,db.execute('SELECT * FROM events WHERE id=?',(event_id,)).fetchone()))

    @app.get('/api/events/export.csv')
    def export_events():
        with connect() as db:
            refresh(db);rows=db.execute('SELECT * FROM events ORDER BY created').fetchall()
            stream=io.StringIO();writer=csv.writer(stream)
            writer.writerow(['event_id','worker_id','status','created_utc','fall_score','source','model_id'])
            for row in rows:writer.writerow([row['id'],row['worker_id'],row['status'],datetime.fromtimestamp(row['created'],timezone.utc).isoformat(),row['score'],row['source'],row['model_id']])
            return Response(stream.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=fall_events.csv'})

    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=5050);parser.add_argument('--no-browser',action='store_true')
    args=parser.parse_args();app=create_app()
    url=f'http://127.0.0.1:{args.port}'
    print('\nFall Detection software demo: '+url+'\nKeep this window open. Press Ctrl+C to stop.\n')
    if not args.no_browser:threading.Timer(1.0,lambda:webbrowser.open(url)).start()
    app.run(host='127.0.0.1',port=args.port,debug=False)
