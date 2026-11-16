import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { useWebcam } from '../../live-demo/useWebcam'
import { useLandmarker } from '../../live-demo/useLandmarker'
import { mouthAlignment } from '../../live-demo/mouthUtils'
import { AdaptiveEvaluation } from '../components/AdaptiveEvaluation'
import { API_BASE } from '../../lib/api'
import { adaptiveApi,feedbackDecision,validSourceWindow,type AdaptiveConfig,type FeedbackRecord,type FeedbackState,type FeedbackTask } from '../../live-demo/adaptiveFeedback'
import '../../styles/adaptive-feedback.css'

interface Draft {id:string;task:FeedbackTask;speaker_id:string;session_id:string;recorded_at:string;capture_kind:string;consent:true;frames:string[];timestamps_ms:number[];original_video?:string}
interface Stats {total:number;confirmed:number;corrected:number;unique_speakers:number;statuses:Record<string,number>;classes:Record<string,number>;growth:Record<string,number>;training_eligible:number;versions:{id:string;task:string;created:string;manifest_sha256:string}[]}
interface Registry {models:Model[];champions:Record<string,{model:string;previous:string|null}>}
interface Model {id:string;task:string;version:string;training_dataset_version:string;deployment_status:string;validation_report:Record<string,unknown>|null;test_report:Record<string,unknown>|null;comparison?:{eligible:boolean;reasons:string[]};cpu_latency_ms:number|null}

export function AdaptiveFeedbackPage() {
  const cam=useWebcam(),lm=useLandmarker()
  const [key,setKey]=useState(''),[config,setConfig]=useState<AdaptiveConfig|null>(null),[tab,setTab]=useState('capture')
  const [task,setTask]=useState<FeedbackTask>('isolated_word'),[speaker,setSpeaker]=useState(''),[session,setSession]=useState<string>(()=>crypto.randomUUID())
  const [consent,setConsent]=useState(false),[state,setState]=useState<FeedbackState>('IDLE'),[quality,setQuality]=useState('Camera is off.')
  const [count,setCount]=useState(0),[draft,setDraft]=useState<Draft|null>(null),[record,setRecord]=useState<FeedbackRecord|null>(null)
  const [correction,setCorrection]=useState(false),[label,setLabel]=useState(''),[error,setError]=useState(''),[notice,setNotice]=useState('')
  const [stats,setStats]=useState<Stats|null>(null),[records,setRecords]=useState<FeedbackRecord[]>([]),[models,setModels]=useState<Registry|null>(null)
  const [index,setIndex]=useState(0),[playing,setPlaying]=useState(false),[preview,setPreview]=useState(''),[busy,setBusy]=useState(false)
  const [filter,setFilter]=useState(''),[taskFilter,setTaskFilter]=useState(''),[search,setSearch]=useState(''),[reason,setReason]=useState('')
  const [assignments,setAssignments]=useState('{}'),[version,setVersion]=useState(''),[job,setJob]=useState(''),[jobResult,setJobResult]=useState<unknown>(null)
  const [replay,setReplay]=useState(''),[demoStep,setDemoStep]=useState(0)
  const [nativePreview,setNativePreview]=useState(''),nativePreviewUrl=useRef('')
  const captureRef=useRef<Draft|null>(null),started=useRef(0),generation=useRef(0),alive=useRef(true),inflight=useRef(false),abort=useRef<AbortController|null>(null)
  const sourceUrl=useRef(''),frameUrl=useRef(''),stateRef=useRef(state);stateRef.current=state
  const nativeRecorder=useRef<{recorder:MediaRecorder;stopped:Promise<Blob>;streams:MediaStream[]}|null>(null)
  const reviewer=config?.role==='reviewer'
  const invoke=<T,>(path:string,body?:unknown)=>adaptiveApi<T>(key,path,body)
  const refresh=async()=>{
    const [s,r,m]=await Promise.all([invoke<Stats>('/stats'),invoke<{records:FeedbackRecord[]}>('/records'),invoke<Registry>('/models')])
    if(alive.current){setStats(s);setRecords(r.records);setModels(m)}
  }
  const action=async(work:()=>Promise<void>)=>{
    if(inflight.current)return
    inflight.current=true;setBusy(true);setError('');setNotice('')
    try{await work()}catch(e){if(alive.current)setError(e instanceof Error?e.message:String(e))}
    finally{inflight.current=false;if(alive.current)setBusy(false)}
  }
  useEffect(()=>{alive.current=true;return()=>{alive.current=false;generation.current++;abort.current?.abort();captureRef.current=null;const native=nativeRecorder.current;if(native?.recorder.state==='recording')native.recorder.stop();native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));URL.revokeObjectURL(sourceUrl.current);URL.revokeObjectURL(frameUrl.current);URL.revokeObjectURL(nativePreviewUrl.current)}},[])
  useEffect(()=>{if(!playing||!draft)return;const next=index+1;const timer=setTimeout(()=>{if(next>=draft.frames.length){setPlaying(false);setIndex(0)}else setIndex(next)},Math.max(15,(draft.timestamps_ms[next]??draft.timestamps_ms[index]+100)-draft.timestamps_ms[index]));return()=>clearTimeout(timer)},[playing,index,draft])
  const finish=async()=>{
    const row=captureRef.current;captureRef.current=null
    if(!row)return
    const token=generation.current,native=nativeRecorder.current;nativeRecorder.current=null
    if(replay)cam.videoRef.current?.pause()
    setState('PROCESSING');setBusy(true);inflight.current=true
    try{
      if(!native)throw new Error('Original video recording was unavailable. Retry this utterance.')
      if(native.recorder.state==='recording')native.recorder.stop()
      const blob=await native.stopped
      if(!blob.size||blob.size>18*1024*1024)throw new Error('Original video is empty or exceeds 18 MiB. Record a shorter utterance.')
      row.original_video=await new Promise<string>((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(String(reader.result));reader.onerror=()=>reject(new Error('Original video could not be read.'));reader.readAsDataURL(blob)})
      validSourceWindow(row.timestamps_ms,row.frames.reduce((n,f)=>n+f.length,0)+row.original_video.length)
      if(!alive.current||generation.current!==token)return
      setDraft(row);setIndex(0);setState('AWAITING_FEEDBACK');setNotice('Review the original bounded video and source frames, then run prediction. No label has been assigned.')
    }catch(e){if(alive.current&&generation.current===token){setError(String(e));setState('REJECTED')}}
    finally{native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));inflight.current=false;if(alive.current)setBusy(false)}
  }
  const finishRef=useRef(finish);finishRef.current=finish
  useEffect(()=>{
    if(lm.status!=='ready'||(!replay&&cam.status!=='live'))return
    const canvas=document.createElement('canvas'),ctx=canvas.getContext('2d');let raf=0,last=0,videoTime=-1
    const loop=(now:number)=>{
      raf=requestAnimationFrame(loop)
      const video=cam.videoRef.current,tracker=lm.landmarkerRef.current
      if(!ctx||!video||!tracker||video.readyState<2||now-last<90||video.currentTime===videoTime)return
      last=now;videoTime=video.currentTime
      try{
        const faces=tracker.detectForVideo(video,now).faceLandmarks
        const aligned=faces.length===1?mouthAlignment(faces[0],video.videoWidth,video.videoHeight):null
        setQuality(aligned?'One face; mouth landmarks detected.':faces.length>1?'Keep exactly one face visible.':'Center your face and mouth in view.')
        const row=captureRef.current
        if(row){
          const elapsed=now-started.current
          if(elapsed>=4800||row.frames.length>=50){finishRef.current();return}
          if(!aligned){captureRef.current=null;const native=nativeRecorder.current;nativeRecorder.current=null;if(native?.recorder.state==='recording')native.recorder.stop();native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));setState('REJECTED');setError('Face or mouth tracking was lost. Retry the complete utterance.');return}
          if(video.videoWidth>1280||video.videoHeight>720)throw new Error('Choose a native camera or replay resolution at most 1280×720.')
          canvas.width=video.videoWidth;canvas.height=video.videoHeight
          ctx.drawImage(video,0,0);const frame=canvas.toDataURL('image/png')
          if(row.frames.reduce((n,f)=>n+f.length,0)+frame.length>85*1024*1024){finishRef.current();return}
          row.frames.push(frame);row.timestamps_ms.push(elapsed);setCount(row.frames.length)
        }
      }catch{if(captureRef.current){captureRef.current=null;const native=nativeRecorder.current;nativeRecorder.current=null;if(native?.recorder.state==='recording')native.recorder.stop();native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));setState('ERROR');setError('Capture processing failed. Retry the camera.')}}
    }
    raf=requestAnimationFrame(loop)
    const watchdog=setInterval(()=>{if(captureRef.current&&performance.now()-started.current>5000)finishRef.current()},100)
    return()=>{cancelAnimationFrame(raf);clearInterval(watchdog)}
  },[cam.status,lm.status,replay])
  useEffect(()=>{if(cam.status==='error'||cam.status==='denied'){captureRef.current=null;const native=nativeRecorder.current;nativeRecorder.current=null;if(native?.recorder.state==='recording')native.recorder.stop();native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));setState('ERROR')}},[cam.status])
  useEffect(()=>{
    if(!record?.media_available||!record.original_video_available||draft){setNativePreview('');return}
    const controller=new AbortController()
    void fetch(`${API_BASE}/api/adaptive/records/${encodeURIComponent(record.id)}/original-video`,{headers:{Authorization:`Bearer ${key}`},signal:controller.signal})
      .then(async r=>{if(!r.ok)throw new Error('Original video unavailable.');return r.blob()})
      .then(blob=>{if(controller.signal.aborted)return;URL.revokeObjectURL(nativePreviewUrl.current);nativePreviewUrl.current=URL.createObjectURL(blob);setNativePreview(nativePreviewUrl.current)})
      .catch(e=>{if(!controller.signal.aborted)setError(String(e))})
    return()=>controller.abort()
  },[record?.id,record?.media_available,record?.original_video_available,draft,key])
  useEffect(()=>{
    if(!record||draft||!record.media_available)return
    const controller=new AbortController()
    void fetch(`${API_BASE}/api/adaptive/records/${encodeURIComponent(record.id)}/frames/${index}`,{headers:{Authorization:`Bearer ${key}`},signal:controller.signal})
      .then(async r=>{if(!r.ok)throw new Error('Frame unavailable.');return r.blob()})
      .then(blob=>{if(controller.signal.aborted)return;URL.revokeObjectURL(frameUrl.current);frameUrl.current=URL.createObjectURL(blob);setPreview(frameUrl.current)})
      .catch(e=>{if(!controller.signal.aborted)setError(String(e))})
    return()=>controller.abort()
  },[record?.id,index,draft,key])
  const reset=()=>{generation.current++;abort.current?.abort();captureRef.current=null;const native=nativeRecorder.current;nativeRecorder.current=null;if(native?.recorder.state==='recording')native.recorder.stop();native?.streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));setDraft(null);setRecord(null);setState('IDLE');setCorrection(false);setLabel('');setIndex(0);setPlaying(false);setPreview('');setCount(0)}
  const begin=()=>void action(async()=>{
    if(!consent||!config||busy||state==='CAPTURING')return
    if(!/^[A-Za-z0-9_-]{1,80}$/.test(speaker)||!/^[A-Za-z0-9_-]{1,80}$/.test(session)){setError('Enter valid, consented speaker and session IDs. Use the same speaker ID in every session.');return}
    reset();setError('');setNotice('')
    const video=cam.videoRef.current
    if(!video||typeof MediaRecorder==='undefined')throw new Error('This browser needs MediaRecorder support for complete video capture.')
    if(replay){video.currentTime=0;await video.play()}
    const replayVideo=video as HTMLVideoElement&{captureStream?:()=>MediaStream}
    const sourceStream=replay?replayVideo.captureStream?.():cam.streamRef.current
    if(!sourceStream?.getVideoTracks().length)throw new Error('Original video capture is unavailable. Use a supported Chrome camera or replay source.')
    const stream=new MediaStream(sourceStream.getVideoTracks().map(t=>t.clone()))
    const streams=replay?[stream,sourceStream]:[stream]
    const mime=['video/webm;codecs=vp8','video/webm','video/mp4'].find(m=>MediaRecorder.isTypeSupported(m))
    if(!mime){streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));throw new Error('No supported WebM/MP4 video recorder is available.')}
    const recorder=new MediaRecorder(stream,{mimeType:mime}),chunks:Blob[]=[]
    const stopped=new Promise<Blob>(resolve=>{recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)};recorder.onstop=()=>resolve(new Blob(chunks,{type:mime.split(';')[0]}));recorder.onerror=()=>{resolve(new Blob());captureRef.current=null;streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));setState('ERROR');setError('Original video capture failed. Retry this utterance.')}})
    try{recorder.start(200)}catch(e){streams.forEach(s=>s.getTracks().forEach(t=>t.stop()));throw e}
    nativeRecorder.current={recorder,stopped,streams};started.current=performance.now()
    captureRef.current={id:crypto.randomUUID(),task,speaker_id:speaker,session_id:session,recorded_at:new Date().toISOString(),capture_kind:replay?'webcam_replay':'physical_webcam',consent:true,frames:[],timestamps_ms:[]}
    setState('CAPTURING')
  })
  const predict=()=>action(async()=>{
    const token=++generation.current;abort.current?.abort();abort.current=new AbortController();setState('PROCESSING')
    try{
      const value=await adaptiveApi<FeedbackRecord>(key,record?`/captures/${record.id}/predict`:'/captures',record?{}:draft,abort.current.signal)
      if(!alive.current||token!==generation.current)return
      setRecord(value);setState(value.prediction?.prediction?'PREDICTED':value.prediction?'UNCERTAIN':'ERROR')
      if(value.prediction_error)setError(value.prediction_error)
      await refresh()
    }catch(e){if(token===generation.current){setState('ERROR');throw e}}
  })
  const submit=(verdict:'correct'|'incorrect')=>action(async()=>{
    if(!record?.prediction||!config)return
    const decision=feedbackDecision(record.prediction,verdict,verdict==='correct'?record.prediction.prediction??'':label,config.correction_labels[record.task])
    const value=await invoke<FeedbackRecord>(`/captures/${record.id}/feedback`,decision)
    setRecord(value);setState('FEEDBACK_SAVED');setNotice('Human feedback saved. Quality checks and reviewer approval are required before dataset inclusion.');await refresh()
  })
  const inspect=(row:FeedbackRecord)=>{reset();setRecord(row);setTab('capture');setState(row.prediction?.prediction?'PREDICTED':'UNCERTAIN')}
  const filtered=records.filter(r=>(!filter||r.status===filter)&&(!taskFilter||r.task===taskFilter)&&(!search||JSON.stringify([r.speaker,r.session,r.label,r.prediction?.candidate,r.prediction?.model_version,r.created]).toLowerCase().includes(search.toLowerCase())))
  return <div className="af-page">
    <header className="ws-page-header"><div><p className="ws-eyebrow">HUMAN VERIFIED LEARNING</p><h1>Adaptive speech lab</h1><p>Record a complete utterance, inspect real model output, and verify the spoken label.</p></div><Link className="ws-btn ws-btn-ghost" to="/demo/live">Live recognition</Link></header>
    <section className="ws-card af-access"><label>Private access key<input type="password" autoComplete="off" value={key} disabled={!!config} onChange={e=>setKey(e.target.value)} placeholder="Collector or reviewer key"/></label><button className="ws-btn ws-btn-primary" disabled={busy||!!config||key.length<32} onClick={()=>void action(async()=>{const c=await invoke<AdaptiveConfig>('/config');setConfig(c);await refresh()})}>Connect</button>{config&&<><span>{config.role} access</span><button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>{reset();cam.stop();setKey('');setConfig(null);setRecords([]);setModels(null);setStats(null);setPreview('')}}>Lock workspace</button></>}</section>
    <p className="af-notice">Current models retain their measured scope. Scores are uncalibrated. Complete A–Z and physical webcam accuracy are unproven.</p>
    {error&&<p role="alert" className="af-error">{error}</p>}{notice&&<p role="status" className="af-notice">{notice}</p>}
    {config&&<>
      <nav className="af-tabs" aria-label="Adaptive lab views">{[['capture','Prediction & feedback'],['dataset','Dataset Feedback Studio'],['models','Models & training'],['demo','Examiner walkthrough']].map(([id,name])=><button key={id} className="ws-btn ws-btn-ghost" aria-pressed={tab===id} disabled={state==='CAPTURING'||busy} onClick={()=>setTab(id)}>{name}</button>)}</nav>
      {stats&&<div className="af-stats">{[['Feedback records',stats.total],['Confirmed',stats.confirmed],['Corrected',stats.corrected],['Pending review',stats.statuses.PENDING_REVIEW??0],['Speakers',stats.unique_speakers],['Approved',stats.training_eligible]].map(([name,n])=><div className="ws-card" key={name}><strong>{n}</strong><span>{name}</span></div>)}</div>}
      {tab==='capture'&&<>
        <div className="af-columns"><section className="ws-card">
          <div className="af-title"><h2>{replay?'Recorded video replay':'Live camera'}</h2><span role="status">{state}</span></div>
          <video ref={cam.videoRef} autoPlay={!replay} muted playsInline src={replay||undefined} onEnded={()=>{if(captureRef.current)finish()}} className="af-camera"/>
          <p aria-live="polite">{quality} {cam.error||lm.error} · {count} captured source frames</p><progress max={50} value={count} aria-label="Capture progress"/>
          <div className="af-actions"><button className="ws-btn ws-btn-primary" disabled={busy||state==='CAPTURING'} onClick={()=>void action(async()=>{URL.revokeObjectURL(sourceUrl.current);setReplay('');if(await cam.start()==='live')await lm.load()})}>Start camera</button><button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>{reset();cam.stop();setQuality('Camera is off.')}}>Stop / discard</button></div>
          <label>Genuine recorded video for replay<input type="file" accept="video/*" disabled={busy||state==='CAPTURING'} onChange={e=>{const file=e.target.files?.[0];if(!file)return;cam.stop();reset();URL.revokeObjectURL(sourceUrl.current);sourceUrl.current=URL.createObjectURL(file);setReplay(sourceUrl.current);void lm.load()}}/></label>
          <p>Replay is always labelled. It uses actual source frames and model inference. No prediction fixtures are used.</p>
          <label>Task<select value={task} disabled={busy||!!draft||!!record||state==='CAPTURING'} onChange={e=>setTask(e.target.value as FeedbackTask)}><option value="isolated_word">Isolated words</option><option value="alphabet_recognition">Alphabet A–Z (experimental)</option></select></label>
          <p>Model vocabulary: {config.tasks[task].join(', ')}</p>
          <div className="af-fields"><label>Consented speaker ID<input value={speaker} disabled={busy||state==='CAPTURING'} onChange={e=>setSpeaker(e.target.value)} placeholder="Stable pseudonymous ID"/></label><label>Session ID<input value={session} disabled={busy||state==='CAPTURING'} onChange={e=>setSession(e.target.value)}/></label></div>
          <label className="af-check"><input type="checkbox" checked={consent} disabled={busy||state==='CAPTURING'} onChange={e=>setConsent(e.target.checked)}/>{config.recording_consent}</label>
          <div className="af-actions"><button className="ws-btn ws-btn-primary" disabled={!consent||busy||lm.status!=='ready'||(!replay&&cam.status!=='live')||state==='CAPTURING'} onClick={begin}>Start utterance</button><button className="ws-btn ws-btn-ghost" disabled={state!=='CAPTURING'} onClick={finish}>Finish utterance</button></div>
          <p>Start before lip movement; finish after the whole word or letter. Lossless sampled frames, maximum five seconds; no audio or unlimited stream.</p>
        </section><section className="ws-card">
          <h2>Review & verify</h2>
          {(draft?.original_video||nativePreview)&&<><h3>Original bounded video</h3><video className="af-source" controls playsInline src={draft?.original_video??nativePreview}/></>}
          {(draft||record?.media_available)&&<><img className="af-source" src={draft?.frames[index]??preview} alt={`Original source frame ${index+1}`}/><label>Source frame {index+1} / {draft?.frames.length??record?.frame_count}<input type="range" min={0} max={(draft?.frames.length??record?.frame_count??1)-1} value={index} onChange={e=>{setPlaying(false);setIndex(Number(e.target.value))}}/></label>{draft&&<button className="ws-btn ws-btn-ghost" onClick={()=>setPlaying(p=>!p)}>{playing?'Pause':'Replay recording'}</button>}</>}
          <button className="ws-btn ws-btn-primary" disabled={busy||(!draft&&!record)||state==='CAPTURING'||!!record?.prediction} onClick={()=>void predict()}>{state==='PROCESSING'?'Processing actual model…':record?'Retry prediction':'Predict reviewed recording'}</button>
          {record?.prediction&&<><div className="af-prediction"><small>{record.task==='isolated_word'?'PREDICTED WORD':'PREDICTED LETTER'}</small><strong>{record.prediction.prediction??'UNCERTAIN'}</strong>{!record.prediction.prediction&&<span>Diagnostic candidate: {record.prediction.candidate}</span>}<span>Score {(record.prediction.confidence*100).toFixed(1)}% · {record.prediction.latency_ms?.toFixed(0)??'unmeasured'} ms</span></div><p>Model: {record.prediction.model_id}<br/><small className="af-hash">Version: {record.prediction.model_version}</small></p><ol>{record.prediction.top_k.map(p=><li key={p.label}>{p.label} <meter min={0} max={1} value={p.score}/> {(p.score*100).toFixed(1)}%</li>)}</ol>
            <h3>Was this prediction correct?</h3><div className="af-actions"><button className="ws-btn ws-btn-primary" disabled={busy||!!record.verdict||!record.prediction.prediction} onClick={()=>void submit('correct')}>Yes, correct</button><button className="ws-btn ws-btn-ghost" disabled={busy||!!record.verdict} onClick={()=>{setCorrection(true);setLabel('');setState('AWAITING_FEEDBACK')}}>No, incorrect</button></div>
            {correction&&<><label>Actual spoken label<select value={label} disabled={busy||!!record.verdict} onChange={e=>setLabel(e.target.value)}><option value="">Choose the actual label</option>{config.correction_labels[record.task].map(l=><option key={l}>{l}</option>)}<option value="UNKNOWN">UNKNOWN / OTHER</option></select></label><button className="ws-btn ws-btn-primary" disabled={busy||!label||!!record.verdict} onClick={()=>void submit('incorrect')}>Save correction</button></>}
          </>}
          {record&&<><p>Dataset status: <strong>{record.status}</strong> · Verified label: {record.label??'not supplied'}</p><div className="af-actions"><button className="ws-btn ws-btn-ghost" disabled={busy||record.status==='WITHDRAWN'} onClick={()=>void action(async()=>{await invoke(`/captures/${record.id}/withdraw`,{});setRecord({...record,status:'WITHDRAWN',media_available:false});setDraft(null);setPreview('');setNotice('Recording withdrawn and source media deleted. Committed lineage, if any, requires model review.');await refresh()})}>Undo / withdraw recording</button>{reviewer&&record.verdict&&<><label>Review reason<input value={reason} onChange={e=>setReason(e.target.value)}/></label>{['APPROVED','REJECTED','QUARANTINED'].map(decision=><button key={decision} className="ws-btn ws-btn-ghost" disabled={busy||reason.trim().length<3} onClick={()=>void action(async()=>{const value=await invoke<FeedbackRecord>(`/records/${record.id}/review`,{decision,reason});setRecord(value);await refresh()})}>{decision}</button>)}</>}</div></>}
          <button className="ws-btn ws-btn-ghost" disabled={busy||state==='CAPTURING'} onClick={reset}>Retry / next utterance</button>
        </section></div>
      </>}
      {tab==='dataset'&&<section className="ws-card"><div className="af-title"><h2>Dataset Feedback Studio</h2><button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>void action(refresh)}>Refresh</button></div><div className="af-fields"><label>Task<select value={taskFilter} onChange={e=>setTaskFilter(e.target.value)}><option value="">All tasks</option><option value="isolated_word">Words</option><option value="alphabet_recognition">Alphabet</option></select></label><label>Feedback status<select value={filter} onChange={e=>setFilter(e.target.value)}><option value="">All statuses</option>{['PENDING_REVIEW','APPROVED','REJECTED','DUPLICATE','QUARANTINED','WITHDRAWN','TRAINING_INCLUDED'].map(s=><option key={s}>{s}</option>)}</select></label><label>Label, speaker, session, model version or date<input value={search} onChange={e=>setSearch(e.target.value)}/></label></div>
        <div className="af-table"><table><thead><tr><th>Recorded</th><th>Task / source</th><th>Prediction → verified label</th><th>Speaker / session</th><th>Status</th><th>Inspect</th></tr></thead><tbody>{filtered.map(r=><tr key={r.id}><td>{new Date(r.created).toLocaleString()}</td><td>{r.task}<br/>{r.capture_kind}</td><td>{r.prediction?.candidate??'unavailable'} → {r.label??'unverified'}</td><td>{r.speaker}<br/><small>{r.session}</small></td><td>{r.status}<br/><small>{r.warnings.join(', ')}</small></td><td><button className="ws-btn ws-btn-ghost" onClick={()=>inspect(r)}>Review source</button></td></tr>)}</tbody></table></div>{!filtered.length&&<p>No matching records. Counters update from saved server records.</p>}
        {stats&&<div className="af-columns"><div><h3>Class distribution</h3>{Object.entries(stats.classes).map(([l,n])=><p key={l}>{l}: {n}</p>)}</div><div><h3>Dataset growth</h3>{Object.entries(stats.growth).map(([d,n])=><p key={d}>{d}: {n} recordings</p>)}<p>Rejected: {stats.statuses.REJECTED??0} · Duplicate: {stats.statuses.DUPLICATE??0} · Quarantined: {stats.statuses.QUARANTINED??0}</p></div></div>}
        <button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>void action(async()=>{const r=await invoke<{records:FeedbackRecord[]}>('/hard-examples');setRecords(r.records);setNotice('Showing reviewable hard examples. Low confidence alone does not prove a wrong label.')})}>Inspect hard examples</button>
      </section>}
      {tab==='models'&&<>
        <section className="ws-card"><h2>Evidence & current champions</h2><p>Offline, replay and physical webcam metrics remain separate. No new physical webcam evaluation or alphabet accuracy benchmark is available.</p>{models&&Object.entries(models.champions).map(([t,c])=><p key={t}><strong>{t}</strong>: {c.model}{reviewer&&<button className="ws-btn ws-btn-ghost" disabled={busy||!c.previous} onClick={()=>void action(async()=>{await invoke(`/models/${t}/rollback`,{});await refresh()})}>Rollback</button>}</p>)}<div className="af-models">{models?.models.map(m=><article className="ws-card" key={m.id}><h3>{m.id}</h3><p>{m.task} · {m.deployment_status}</p><p className="af-hash">Version: {m.version}<br/>Dataset: {m.training_dataset_version}</p><p>Validation accuracy: {typeof m.validation_report?.top1_accuracy==='number'?`${(m.validation_report.top1_accuracy*100).toFixed(2)}%`:'not measured'}</p><p>Macro F1: {String(m.validation_report?.macro_f1??m.validation_report?.macro_f1_usable??'not measured')}</p><p>CPU: {m.cpu_latency_ms?.toFixed(0)??'unmeasured'} ms (see report scope)</p>{m.comparison&&<><h4>Champion / challenger decision</h4><p>{m.comparison.eligible?'Validation gate passed':'Challenger not eligible'}</p><ul>{m.comparison.reasons.map(r=><li key={r}>{r}</li>)}</ul></>}<details><summary>Evaluation evidence</summary><pre>{JSON.stringify({validation:m.validation_report,test:m.test_report},null,2)}</pre></details>{reviewer&&m.deployment_status==='challenger'&&<><button className="ws-btn ws-btn-ghost" disabled={busy||!!m.test_report} onClick={()=>void action(async()=>{await invoke(`/models/${m.id}/final-evaluation`,{});setNotice('Final evaluation requested once. Refresh to inspect saved evidence.')})}>Freeze and evaluate held-out test once</button><button className="ws-btn ws-btn-primary" disabled={busy||!m.comparison?.eligible||!m.test_report} onClick={()=>void action(async()=>{await invoke(`/models/${m.id}/deploy`,{approved:true});await refresh()})}>Explicitly approve champion deployment</button></>}</article>)}</div></section>
        {reviewer&&<section className="ws-card"><h2>Controlled batch training</h2><p>Only approved known-label samples enter a version. Assign each genuine speaker to one partition across all sessions. Existing evaluation speakers and missing classes block training. HELP, I, AM and UNKNOWN need separate protocols.</p><label>Dataset task<select value={task} onChange={e=>setTask(e.target.value as FeedbackTask)}><option value="isolated_word">Eight words</option><option value="alphabet_recognition">Alphabet A–Z</option></select></label><label>Speaker assignments (JSON)<textarea rows={5} value={assignments} onChange={e=>setAssignments(e.target.value)} placeholder={'{"speakerA":"train","speakerB":"valid","speakerC":"test"}'}/></label><button className="ws-btn ws-btn-primary" disabled={busy} onClick={()=>void action(async()=>{const v=await invoke<{id:string}>('/datasets',{task,assignments:JSON.parse(assignments)});setVersion(v.id);setNotice('Immutable dataset version saved. Structural eligibility is not proof of sufficient data.');await refresh()})}>Freeze reviewed dataset version</button><label>Frozen dataset version<select value={version} onChange={e=>setVersion(e.target.value)}><option value="">Select a version</option>{stats?.versions.map(v=><option key={v.id} value={v.id}>{v.task}: {v.id}</option>)}</select></label><button className="ws-btn ws-btn-primary" disabled={busy||!version} onClick={()=>void action(async()=>{const j=await invoke<{job_id:string}>('/training/jobs',{dataset_version:version});setJob(j.job_id);setNotice('Batch job queued. Production checkpoint is preserved.')})}>Train challenger in background</button><label>Training job ID<input value={job} onChange={e=>setJob(e.target.value)}/></label><button className="ws-btn ws-btn-ghost" disabled={busy||!job} onClick={()=>void action(async()=>{setJobResult(await invoke(`/training/jobs/${job}`));await refresh()})}>Read actual job status</button>{jobResult!=null&&<pre>{JSON.stringify(jobResult,null,2)}</pre>}<button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>void action(async()=>{const r=await invoke<{expired:number}>('/retention/expire',{});setNotice(`${r.expired} uncommitted expired recordings removed.`);await refresh()})}>Run 30-day retention maintenance</button></section>}
      </>}
      {tab==='demo'&&<section className="ws-card"><h2>External examiner walkthrough</h2><p>Use a physical webcam or a genuine video file. The input source is labelled throughout; recognition uses the real backend.</p><ol>{['Open Prediction & feedback and connect camera or choose recorded video replay.','Obtain consent; enter stable speaker and session IDs.','Start and finish a complete utterance; inspect the original frame timeline.','Run the actual model and inspect status, top candidates, score, version and latency.','Confirm the accepted prediction or select the actual corrected label.','Show the saved feedback record and updated Dataset Feedback Studio counters.','A reviewer inspects source evidence and approves eligible data.','Freeze a complete speaker-independent version and request batch training.','Review paired validation, risk/coverage, class metrics and resource results.','Keep a rejected challenger separate. Final test evaluation is one-time; deployment requires evidence and explicit reviewer approval.'].map((s,i)=><li key={s} aria-current={demoStep===i?'step':undefined} className={demoStep===i?'af-current':''}>{s}</li>)}</ol><div className="af-actions"><button className="ws-btn ws-btn-ghost" disabled={demoStep===0} onClick={()=>setDemoStep(s=>s-1)}>Previous step</button><button className="ws-btn ws-btn-primary" disabled={demoStep===9} onClick={()=>setDemoStep(s=>s+1)}>Next step</button><button className="ws-btn ws-btn-ghost" onClick={()=>setTab('capture')}>Open recording workflow</button></div><p>No training progress or accuracy is simulated. Empty datasets and unavailable models remain visibly unavailable.</p></section>}
      {tab==='models'&&models?.models.map(m=><AdaptiveEvaluation key={m.id} id={m.id} dataset={m.training_dataset_version} report={m.validation_report}/>)}
    </>}
  </div>
}
