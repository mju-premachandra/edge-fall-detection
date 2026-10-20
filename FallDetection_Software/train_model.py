"""Reproducible prototype training; test people are opened only after settings freeze."""
from pathlib import Path
import argparse, json, hashlib, platform, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, fbeta_score
from core import SENSORS, FEATURES, POLICY, STEP_SECONDS, windows, score_tree

ROOT = Path(__file__).resolve().parent

def dataset(part):
    frames, rows = {}, []
    for path in sorted(part.glob("*_belt.csv")):
        frame = pd.read_csv(path)
        frames[path.stem] = frame
        for win in windows(frame, require_labels=True):
            rows.append({"recording": path.stem, "subject": int(frame.subject_id.iloc[0]),
                         "end_s": win["end_s"], "target": win["target"], "features": win["features"]})
    if not rows:
        raise RuntimeError(f"No valid labelled windows in {part}")
    return frames, rows, np.stack([r["features"] for r in rows]), np.array([r["target"] for r in rows])

def metrics(y, prediction):
    precision, recall, f1, _ = precision_recall_fscore_support(y, prediction, average="binary", zero_division=0)
    return {"precision":float(precision), "recall":float(recall), "f1":float(f1),
            "f2":float(fbeta_score(y, prediction, beta=2, zero_division=0)),
            "confusion_matrix":confusion_matrix(y,prediction,labels=[0,1]).tolist(),
            "windows":int(len(y)), "fall_windows":int(y.sum()),
            "note":"Window-level metrics on overlapping causal windows; not independent event-level estimates."}

def rank(y, predictions):
    result=metrics(y,predictions)
    return (result["f2"],result["precision"])

def export_portable(clf, threshold):
    tree=clf.tree_
    values=tree.value[:,0,:]
    fall_index=list(clf.classes_).index(1)
    scores=values[:,fall_index]/values.sum(axis=1)
    return {"model_id":"fall-tree-v1", "algorithm":"DecisionTreeClassifier", "feature_names":FEATURES,
            "window_policy":POLICY, "decision_threshold":float(threshold),
            "score_note":"Leaf class fraction after balanced class weighting; not a calibrated probability of injury or a real-world fall.",
            "tree":{"children_left":tree.children_left.tolist(),"children_right":tree.children_right.tolist(),
                    "feature":tree.feature.tolist(),"threshold":tree.threshold.tolist(),"fall_score":scores.tolist()},
            "training":{"sklearn":sklearn.__version__,"numpy":np.__version__,"python":platform.python_version(),
                        "subjects":[1,2,3],"validation_subjects":[4],"test_subjects":[5,6],
                        "settings":clf.get_params(),"selected_with":"Validation F2, then precision; simpler depth wins ties."}}

def event_metrics(frames, rows, predictions):
    """Fixed one-to-one event matching, before haptic cancellation."""
    detected=truth_total=nonfall_alarms=0; exposure=0.0; latencies=[]
    table=pd.DataFrame([{k:v for k,v in row.items() if k!='features'} for row in rows])
    table['prediction']=predictions
    for name, frame in frames.items():
        part=table[table.recording.eq(name)]
        positive=part.loc[part.prediction.eq(1),'end_s'].to_numpy(float)
        groups=[]
        for t in positive:
            if not groups or t-groups[-1][-1]>2.0:groups.append([float(t)])
            else:groups[-1].append(float(t))
        times=frame.time_s.to_numpy(float)
        labels=frame.frame_activity_id.to_numpy()
        mask=np.isin(labels,[1,2,3,4,5])
        edges=np.diff(np.r_[False,mask,False].astype(int))
        intervals=[(float(times[a]),float(times[b-1])) for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0])]
        used=set()
        for start,end in intervals:
            truth_total+=1
            for j,g in enumerate(groups):
                hits=[t for t in g if start<=t<=end+1.0]
                if j not in used and hits:
                    detected+=1;used.add(j);latencies.append(min(hits)-start);break
        if int(frame.trial_activity_id.iloc[0])>=6:
            nonfall_alarms+=len(groups)
            exposure+=sum(min(STEP_SECONDS,max(0.0,float(times[-1])-float(t))) for t in part.end_s)
    return {"labelled_fall_intervals":truth_total,"detected_fall_intervals":detected,
            "event_recall":detected/truth_total if truth_total else None,
            "false_events_in_nonfall_recordings":nonfall_alarms,"scored_nonfall_seconds":exposure,
            "false_events_per_scored_nonfall_hour":nonfall_alarms*3600/exposure if exposure else None,
            "median_delay_from_tagged_onset_s":float(np.median(latencies)) if latencies else None,
            "protocol":"Merge positive decisions within 2 s. Match alarm groups one-to-one to a known fall-tag interval through 1 s after its end. Exposure sums up to 0.25 s following each eligible window endpoint in non-fall trials. This is short laboratory exposure, before cancellation, and excludes unscored time."}

def train(data_dir):
    (ROOT/'model').mkdir(exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
    train_frames,train_rows,xtrain,ytrain=dataset(data_dir/'train')
    val_frames,val_rows,xval,yval=dataset(data_dir/'validation')
    assert {r['subject'] for r in train_rows}=={1,2,3}
    assert {r['subject'] for r in val_rows}=={4}
    candidates=[];best=None
    for depth in [3,4,5]:
        for leaf in [5,15]:
            clf=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=leaf,class_weight='balanced',random_state=42)
            clf.fit(xtrain,ytrain)
            val_scores=clf.predict_proba(xval)[:,list(clf.classes_).index(1)]
            for cutoff in [0.35,0.5,0.65]:
                prediction=(val_scores>=cutoff).astype(int)
                m=metrics(yval,prediction)
                candidates.append({'depth':depth,'min_samples_leaf':leaf,'cutoff':cutoff,**m})
                selection=(*rank(yval,prediction),-depth,-leaf)
                if best is None or selection>best[0]:best=(selection,clf,cutoff)
    _,clf,cutoff=best
    # Choose a simple acceleration-peak baseline using validation only.
    baseline_grid=sorted(set([1.5,2.0,2.5,3.0,3.5,4.0,*np.quantile(xtrain[:,3],[.8,.9,.95,.97,.99]).tolist()]))
    baseline=max(baseline_grid,key=lambda t:(*rank(yval,xval[:,3]>=t),t))
    model=export_portable(clf,cutoff)
    model['baseline']={'rule':'acc_max >= threshold_g','threshold_g':float(baseline)}
    # Settings are frozen and written BEFORE loading held-out test people.
    (ROOT/'model/trained_tree.json').write_text(json.dumps(model,indent=2))
    (ROOT/'model/decision_rules.txt').write_text(export_text(clf,feature_names=FEATURES,decimals=6))
    frozen_hash=hashlib.sha256((ROOT/'model/trained_tree.json').read_bytes()).hexdigest()
    test_frames,test_rows,xtest,ytest=dataset(data_dir/'test')
    assert {r['subject'] for r in test_rows}=={5,6}
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'model_sha256':frozen_hash,
            'selection':'Maximise validation window F2; precision then smaller depth then leaf size break ties. No settings use test results.',
            'subjects':{'train':[1,2,3],'validation':[4],'test':[5,6]},'window_policy':POLICY,
            'selected':{'depth':clf.max_depth,'min_samples_leaf':clf.min_samples_leaf,'fall_score_threshold':cutoff,'baseline_peak_g':float(baseline)},
            'candidates':candidates,'splits':{}}
    for name,frames,rows,x,y in [('train',train_frames,train_rows,xtrain,ytrain),('validation',val_frames,val_rows,xval,yval),('test',test_frames,test_rows,xtest,ytest)]:
        scores=clf.predict_proba(x)[:,list(clf.classes_).index(1)]
        portable=np.array([score_tree(model,row) for row in x])
        assert np.allclose(scores,portable,rtol=0,atol=1e-12),'Portable model differs from sklearn'
        pred=(scores>=cutoff).astype(int);base=(x[:,3]>=baseline).astype(int)
        report['splits'][name]={'recordings':len(frames),'model':metrics(y,pred),'baseline':metrics(y,base),
                                'model_events':event_metrics(frames,rows,pred),'baseline_events':event_metrics(frames,rows,base)}
    report['limits']=['Six participants and one repetition per activity; only two held-out participants.',
                      'Laboratory data; not validated for construction work or unsupervised use.',
                      'Unknown labels and ineligible windows are excluded; irregular timing and gaps reduce coverage.',
                      'A leaf score is not a calibrated real-world probability.',
                      'Replay and manual synthetic input do not validate live sensors, physical haptics or on-device timing.',
                      'Reusing held-out recordings for later model choices would turn them into development data; use new people for a new final test.']
    (ROOT/'reports/training_report.json').write_text(json.dumps(report,indent=2))
    lines=['# Model evaluation','',f"Model trained on participants 1–3; selected on 4; tested on 5–6.",
           f"Settings frozen before loading test data. Model SHA-256: `{frozen_hash}`.",'',
           '| Split | Windows | Model precision | Model recall | Model F1 | Baseline F1 |',
           '|---|---:|---:|---:|---:|---:|']
    for split,p in report['splits'].items():
        m,b=p['model'],p['baseline'];lines.append(f"| {split} | {m['windows']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {b['f1']:.3f} |")
    lines+=['','These are window metrics on overlapping windows. Event recall and scored non-fall exposure are reported separately in training_report.json.','',
            '## Fixed input protocol',json.dumps(POLICY,indent=2),'','## Limits',*['- '+x for x in report['limits']]]
    (ROOT/'reports/MODEL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'selected':report['selected'],'validation':report['splits']['validation']['model'],
                      'test':report['splits']['test']['model'],'test_events':report['splits']['test']['model_events']},indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=ROOT/'data/processed')
    args=parser.parse_args();train(args.data)
