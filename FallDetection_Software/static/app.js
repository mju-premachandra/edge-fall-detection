'use strict';
const $ = id => document.getElementById(id);
const state = {tab:'recording', analysis:null, index:0, playing:false, timer:null, events:[], eventPollAt:0, alertRun:null, opening:false, model:null};
const esc = value => String(value).replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const percent = n => `${(n*100).toFixed(1)}%`;

async function api(path, options={}) {
  const start=performance.now();
  let response;
  try {response=await fetch(path, options);} catch(error) {
    $('connection').className='connection bad';$('connection').querySelector('span').textContent='Backend unavailable';
    throw new Error('Cannot reach the backend. Keep app.py running, then try again.');
  }
  const data=await response.json();
  $('connection').className='connection good';$('connection').querySelector('span').textContent='Backend connected';
  if(options.method==='POST') {
    $('request-status').textContent=`POST ${path} · ${response.status} · ${Math.round(performance.now()-start)} ms`;
    $('request-dot').style.background=response.ok?'#159d79':'#bb4053';
  }
  if(!response.ok) throw new Error(data.error||`Request failed (${response.status}).`);
  return data;
}
function post(path,body){return api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});}
function showError(message){$('error').textContent=message;$('error').classList.remove('hidden');}
function clearError(){$('error').classList.add('hidden');}
function stopPlayback(){state.playing=false;if(state.timer)clearTimeout(state.timer);state.timer=null;$('play').textContent='▶ Replay predictions';$('pause').disabled=true;}

document.querySelectorAll('.tab').forEach(button=>button.addEventListener('click',()=>{
  state.tab=button.dataset.tab;
  document.querySelectorAll('.tab').forEach(b=>{b.classList.toggle('selected',b===button);b.setAttribute('aria-selected',String(b===button));});
  ['recording','upload','manual'].forEach(tab=>$(`tab-${tab}`).classList.toggle('hidden',tab!==state.tab));
  clearError();
}));
$('csv-file').addEventListener('change',()=>{$('file-name').textContent=$('csv-file').files[0]?.name||'Choose a sensor CSV';});

function updateRecording(){
  const selected=$('recording').selectedOptions[0];
  $('download-sample').href=`/api/samples/${encodeURIComponent($('recording').value)}/csv`;
  $('recording-meta').textContent=selected?.dataset.meta||'Real UP-Fall recording';
}
$('recording').addEventListener('change',updateRecording);
for(const kind of ['walking','fall']) $('example-'+kind).addEventListener('click',async()=>{
  try {clearError();const example=await api(`/api/examples/${kind}`);$('manual-csv').value=example.csv;}
  catch(error){showError(error.message);}
});
$('append-reading').addEventListener('click',()=>{
  const fields=['acc_x_g','acc_y_g','acc_z_g','gyro_x_deg_s','gyro_y_deg_s','gyro_z_deg_s'];
  const values=fields.map(name=>Number($(name).value));
  if(!values.every(Number.isFinite)){showError('Enter six numeric readings.');return;}
  const header='time_s,'+fields.join(',');
  let lines=$('manual-csv').value.trim().split(/\r?\n/).filter(Boolean);
  if(!lines.length)lines=[header];
  const previous=lines.length>1?Number(lines[lines.length-1].split(',')[0]):-.05;
  if(!Number.isFinite(previous)){showError('The last timestamp is invalid.');return;}
  lines.push([(previous+.05).toFixed(6),...values].join(','));
  $('manual-csv').value=lines.join('\n')+'\n';
});

$('analyze').addEventListener('click',async()=>{
  clearError();stopPlayback();
  const duration=Number($('cancellation-time').value);
  if(!Number.isFinite(duration)||duration<1||duration>120){showError('Set cancellation time between 1 and 120 seconds.');return;}
  if(!$('worker-id').value.trim()){showError('Enter a demo worker ID.');return;}
  $('analyze').disabled=true;$('model-choice').disabled=true;$('analyze').innerHTML='Analyzing input… <span>↻</span>';
  try {
    let result;
    const model_id=state.model.model_id;
    if(state.tab==='recording')result=await post('/api/analyze',{sample_id:$('recording').value,model_id});
    else if(state.tab==='upload') {
      const file=$('csv-file').files[0];if(!file)throw new Error('Choose a CSV file first.');
      const body=new FormData();body.append('file',file);body.append('model_id',model_id);result=await api('/api/analyze',{method:'POST',body});
    } else result=await post('/api/analyze',{csv:$('manual-csv').value,model_id});
    state.analysis=result;state.alertRun=null;state.index=0;
    const firstFall=result.predictions.findIndex(p=>p.suspected_fall);
    state.index=firstFall<0?0:firstFall;
    $('chart-empty').classList.add('hidden');$('play').disabled=false;
    await showPrediction(state.index,true);
    $('prediction-note').textContent=firstFall<0?`No fall predicted in ${result.predictions.length} eligible windows.`:`First suspected-fall window at ${result.predictions[firstFall].time_s.toFixed(2)} s; ${result.suspected_fall_windows} flagged windows.`;
    $('api-response').textContent=JSON.stringify({model_id:result.model_id,run_id:result.run_id,input_rows:result.input_rows,source:result.source,eligible_windows:result.predictions.length,suspected_fall_windows:result.suspected_fall_windows,displayed_prediction:result.predictions[state.index]},null,2);
  } catch(error){showError(error.message);}
  finally{$('analyze').disabled=!state.model;$('model-choice').disabled=false;$('analyze').innerHTML='Analyze input <span>→</span>';}
});

async function showPrediction(index,openAlert){
  if(!state.analysis)return;
  const p=state.analysis.predictions[index];state.index=index;
  $('model-id').textContent=state.analysis.model_id;
  $('prediction-banner').className='prediction-banner '+(p.suspected_fall?'danger':'safe');
  $('prediction-icon').textContent=p.suspected_fall?'!':'✓';
  $('prediction-label').textContent=p.suspected_fall?'Suspected fall':'No fall predicted';
  $('prediction-note').textContent=`Window ${p.start_s.toFixed(2)}–${p.time_s.toFixed(2)} s · ${p.samples} readings · ${state.analysis.source}`;
  $('fall-score').textContent=percent(p.fall_score);$('score-bar').style.width=percent(p.fall_score);
  $('score-bar').style.background=p.suspected_fall?'#bb4053':'#087f78';
  $('playback-time').textContent=p.time_s.toFixed(2)+' s';
  drawChart();
  if(openAlert&&p.suspected_fall&&state.alertRun!==state.analysis.run_id&&!state.opening){
    state.opening=true;
    try {
      await post('/api/events',{decision_id:p.decision_id,worker_id:$('worker-id').value.trim(),cancellation_seconds:Number($('cancellation-time').value)});
      state.alertRun=state.analysis.run_id;await pollEvents();
    }catch(error){showError(error.message);}
    finally{state.opening=false;}
  }
}
$('play').addEventListener('click',()=>{
  if(!state.analysis)return;
  stopPlayback();state.index=0;state.playing=true;$('pause').disabled=false;$('play').textContent='↻ Restart replay';
  const tick=async()=>{
    if(!state.playing)return;
    try{await showPrediction(state.index,true);}catch(error){showError(error.message);stopPlayback();return;}
    if(state.index>=state.analysis.predictions.length-1){stopPlayback();return;}
    const delay=(state.analysis.predictions[state.index+1].time_s-state.analysis.predictions[state.index].time_s)*1000/Number($('speed').value);
    state.index++;state.timer=setTimeout(tick,delay);
  };tick();
});
$('pause').addEventListener('click',stopPlayback);

function drawChart(){
  const canvas=$('signal-chart'),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const w=rect.width,h=rect.height,pad={l:33,r:12,t:14,b:25};
  ctx.clearRect(0,0,w,h);
  if(!state.analysis)return;
  const trace=state.analysis.trace,maxTime=Math.max(.001,trace[trace.length-1].time_s);
  const maxY=Math.max(1.6,...trace.map(p=>p.magnitude_g))*1.12;
  const x=t=>pad.l+t/maxTime*(w-pad.l-pad.r),y=a=>h-pad.b-a/maxY*(h-pad.t-pad.b);
  ctx.font='9px monospace';ctx.fillStyle='#8c9d99';ctx.lineWidth=1;
  for(let i=0;i<=3;i++){
    const value=maxY*i/3,yy=y(value);ctx.strokeStyle='#edf1ef';ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();ctx.fillText(value.toFixed(1),2,yy+3);
    const t=maxTime*i/3;ctx.fillText(t.toFixed(1)+'s',x(t)-8,h-6);
  }
  ctx.setLineDash([3,4]);ctx.strokeStyle='#b8c6bf';ctx.beginPath();ctx.moveTo(pad.l,y(1));ctx.lineTo(w-pad.r,y(1));ctx.stroke();ctx.setLineDash([]);
  ctx.strokeStyle='#7561b3';ctx.lineWidth=1.3;ctx.beginPath();trace.forEach((p,i)=>{if(i===0)ctx.moveTo(x(p.time_s),y(p.magnitude_g));else ctx.lineTo(x(p.time_s),y(p.magnitude_g));});ctx.stroke();
  const p=state.analysis.predictions[state.index];
  if(p){ctx.fillStyle=p.suspected_fall?'#bb405312':'#087f7810';ctx.fillRect(x(p.start_s),pad.t,x(p.time_s)-x(p.start_s),h-pad.t-pad.b);ctx.strokeStyle=p.suspected_fall?'#bb4053':'#087f78';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(x(p.time_s),pad.t);ctx.lineTo(x(p.time_s),h-pad.b);ctx.stroke();}
}
window.addEventListener('resize',drawChart);

let polling=false;
async function pollEvents(){
  if(polling)return;polling=true;
  try {
    const result=await api('/api/events');state.events=result.events;state.eventPollAt=performance.now();
    $('event-count').textContent=state.events.length;
    if(!state.events.length)$('events-body').innerHTML='<tr><td colspan="7" class="empty-row">Alerts will appear here after a suspected-fall prediction.</td></tr>';
    else $('events-body').innerHTML=state.events.map(e=>`<tr><td>${esc(new Date(e.created*1000).toLocaleTimeString())}</td><td>${esc(e.worker_id)}</td><td>${esc(e.source)}</td><td>${esc(e.model_id)}</td><td>${percent(e.score)}</td><td><span class="status-pill ${esc(e.status)}">${esc(e.status)}</span></td><td>${e.status==='escalated'?`<button class="ack-row" data-ack="${esc(e.id)}">Acknowledge</button>`:e.status==='acknowledged'?'Received':e.status==='cancelled'?'Not escalated':'Waiting'}</td></tr>`).join('');
    document.querySelectorAll('[data-ack]').forEach(b=>b.addEventListener('click',()=>acknowledge(b.dataset.ack)));
    renderAlert();
  }catch(error){console.warn(error.message);}
  finally{polling=false;}
}
function currentEvent(){return state.events.find(e=>e.status==='pending')||state.events[0];}
function renderAlert(){
  const e=currentEvent();if(!e)return;
  $('alert-panel').className='alert-panel '+e.status;
  $('cancel-alert').classList.toggle('hidden',e.status!=='pending');$('ack-alert').classList.toggle('hidden',e.status!=='escalated');
  const titles={pending:'Cancellation window open',cancelled:'Alert cancelled',escalated:'Alert escalated to the local receiver',acknowledged:'Receipt acknowledged'};
  $('alert-title').textContent=titles[e.status]||e.status;
  const notes={pending:`${e.worker_id} · Visual haptic prompt only. Cancel this alert if appropriate.`,cancelled:'The cancellation was logged. This event will not escalate.',escalated:'The cancellation time expired. The local backend has recorded the alert.',acknowledged:'The local receiver acknowledged this event. Its history remains saved.'};
  $('alert-note').textContent=`${e.model_id} · ${notes[e.status]}`;$('countdown').textContent=e.status==='pending'?Math.max(0,e.remaining_seconds-(performance.now()-state.eventPollAt)/1000).toFixed(1)+' s':e.status==='cancelled'?'✓':e.status==='escalated'?'!':'✓';
}
$('cancel-alert').addEventListener('click',async()=>{
  const event=currentEvent();if(!event)return;
  try{clearError();await post(`/api/events/${event.id}/cancel`,{});await pollEvents();}catch(error){showError(error.message);await pollEvents();}
});
async function acknowledge(id){try{clearError();await post(`/api/events/${id}/acknowledge`,{});await pollEvents();}catch(error){showError(error.message);}}
$('ack-alert').addEventListener('click',()=>{const e=currentEvent();if(e)acknowledge(e.id);});

function renderModelMetadata(model){
  state.model=model;
  const isForest=model.algorithm==='RandomForestClassifier';
  const isSvm=model.algorithm==='SVC';
  const isComparison=isForest||isSvm;
  $('model-id').textContent=model.model_id;
  $('threshold-label').textContent='Decision threshold '+percent(model.threshold);
  $('threshold-marker').style.left=percent(model.threshold);
  $('model-choice-note').textContent=isSvm?'Uses scaled features and an RBF support vector machine. The displayed score is a transformed margin, not a probability. Results below reuse subjects 5–6.':isForest?'Uses 100 trained trees. Results below are an exploratory comparison on reused subjects 5–6.':'Uses the original trained decision tree. Select a model, then analyze the same input to compare.';
  $('evaluation-tag').textContent=isComparison?'REUSED DATA · EXPLORATORY':'SMALL LABORATORY SUBSET';
  $('windows-caption').textContent=isComparison?'reused comparison windows':'held-out windows';
  $('recall-caption').textContent=isComparison?'comparison window recall':'test window recall';
  $('precision-caption').textContent=isComparison?'comparison window precision':'test window precision';
  $('evaluation-note').textContent=`${model.label} · Trained on subjects 1–3; settings selected on subject 4. ${model.evaluation_status} This prototype produces false alarms and has not been validated for construction-worker use.`;
  const test=model.evaluation.splits.test,val=model.evaluation.splits.validation,e=test.model_events;
  $('metric-windows').textContent=test.model.windows.toLocaleString();
  $('metric-recall').textContent=percent(test.model.recall);
  $('metric-precision').textContent=percent(test.model.precision);
  const depth=model.training.settings.max_depth??'unlimited';
  const settingsText=isSvm?`RBF kernel; C ${model.training.settings.C}; gamma ${model.training.settings.gamma}; ${model.training.support_vector_count} support vectors; scaling fit on training subjects only`:`${isForest?`${model.training.settings.n_estimators} trees; `:''}depth ${depth}`;
  $('model-report').innerHTML=`<h4>${isComparison?'Exploratory comparison on reused subjects 5–6':'Original held-out results'}</h4><table><thead><tr><th>Method</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody><tr><td>${esc(model.label)}</td><td>${percent(test.model.precision)}</td><td>${percent(test.model.recall)}</td><td>${test.model.f1.toFixed(3)}</td></tr><tr><td>${esc(model.comparison_label)}</td><td>${percent(test.baseline.precision)}</td><td>${percent(test.baseline.recall)}</td><td>${test.baseline.f1.toFixed(3)}</td></tr></tbody></table><p>Event matching: ${e.detected_fall_intervals}/${e.labelled_fall_intervals} labelled fall intervals detected. ${e.false_events_in_nonfall_recordings} false event groups in ${(e.scored_nonfall_seconds/60).toFixed(2)} minutes of scored non-fall exposure, before cancellation. This short exposure cannot establish an operational false-alarm rate.</p><p>${esc(settingsText)}; ${model.feature_names.length} sensor-derived features; 1-second causal windows. Validation F1: ${val.model.f1.toFixed(3)}. Settings were frozen before the reported evaluation.</p><p>${esc(model.score_note)}</p><h4>Practical limits</h4><ul>${model.evaluation.limits.map(v=>'<li>'+esc(v)+'</li>').join('')}</ul>`;
}

async function selectModel(modelId){
  const previous=state.model?.model_id;
  $('analyze').disabled=true;$('model-choice').disabled=true;stopPlayback();
  try {
    const model=await api('/api/model?model_id='+encodeURIComponent(modelId));
    renderModelMetadata(model);
    state.analysis=null;state.index=0;state.alertRun=null;
    $('prediction-banner').className='prediction-banner';$('prediction-icon').textContent='◎';
    $('prediction-label').textContent='Ready for input';
    $('prediction-note').textContent=`${model.label} selected. Click Analyze input to run this model.`;
    $('fall-score').textContent='—';$('score-bar').style.width='0%';
    $('chart-empty').classList.remove('hidden');$('play').disabled=true;
    $('playback-time').textContent='— s';$('api-response').textContent='No prediction request for the selected model yet.';
    drawChart();
  }catch(error){if(previous)$('model-choice').value=previous;throw error;}
  finally{$('analyze').disabled=!state.model;$('model-choice').disabled=false;}
}
$('model-choice').addEventListener('change',async()=>{
  try{clearError();await selectModel($('model-choice').value);}catch(error){showError(error.message);}
});

async function start(){
  try {
    await api('/api/health');
    const samples=await api('/api/samples');
    $('recording').innerHTML=samples.samples.map(s=>`<option value="${esc(s.id)}" data-meta="${esc('Subject '+s.subject+' · '+s.split+' participant · trial 1')}">${esc(s.label)}</option>`).join('');
    $('recording').value='Subject1Activity6Trial1_belt';updateRecording();
    const catalog=await api('/api/models');
    $('model-choice').innerHTML=catalog.models.map(m=>`<option value="${esc(m.id)}">${esc(m.label)}</option>`).join('');
    $('model-choice').value=catalog.default_model_id;
    await selectModel(catalog.default_model_id);
    if(catalog.messages.length)$('model-choice-note').textContent+=' '+catalog.messages.join(' ');
    await pollEvents();
  }catch(error){showError(error.message);}
}
setInterval(pollEvents,800);setInterval(renderAlert,100);start();
