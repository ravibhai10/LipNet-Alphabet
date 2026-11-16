import { useEffect,useRef,useState } from 'react'
import { Link } from 'react-router-dom'
import { useWebcam } from '../../live-demo/useWebcam'
import { useLandmarker } from '../../live-demo/useLandmarker'
import { mouthAlignment } from '../../live-demo/mouthUtils'
import { API_BASE,requestWordPrediction,requestPrediction } from '../../lib/api'
import { digest } from '../../live-demo/wordCollection'
import { ACCEPTANCE_CASES,checkSpeakerAssignment,checkStudioRecording,downloadJson,measuredFps,verifyPlayback,
  type StudioConfig,type StudioInventory,type StudioRecording,type StudioSplit,type StudioTask } from '../../live-demo/recordingStudio'
import '../../styles/word-collection.css'

async function api(path:string,init?:RequestInit) {
  const response=await fetch(`${API_BASE}/api/studio${path}`,init)
  const value=await response.json()
  if (!response.ok) throw new Error(typeof value.detail==='string'?value.detail:'Studio request failed.')
  return value
}

export function RecordingStudioPage() {
  const cam=useWebcam(),lm=useLandmarker()
  const [config,setConfig]=useState<StudioConfig|null>(null),[inventory,setInventory]=useState<StudioInventory|null>(null)
  const [task,setTask]=useState<StudioTask>('isolated_word'),[label,setLabel]=useState('choose')
  const [speaker,setSpeaker]=useState('participant01'),[session,setSession]=useState(()=>`session-${Date.now()}`),[split,setSplit]=useState<StudioSplit>('train')
  const [consent,setConsent]=useState(false),[confirmed,setConfirmed]=useState(false),[kind,setKind]=useState<StudioRecording['capture_kind']>('physical_webcam')
  const [phase,setPhase]=useState<'ready'|'capturing'|'review'|'saving'|'processing'>('ready')
  const [draft,setDraft]=useState<StudioRecording|null>(null),[pending,setPending]=useState<StudioRecording[]>([])
  const [error,setError]=useState<string|null>(null),[notice,setNotice]=useState<string|null>(null),[quality,setQuality]=useState('Camera is off.')
  const [usable,setUsable]=useState(false),[faceCount,setFaceCount]=useState(0),[count,setCount]=useState(0),[reviewIndex,setReviewIndex]=useState(0),[playing,setPlaying]=useState(false)
  const [spoken,setSpoken]=useState(''),[lighting,setLighting]=useState('normal'),[distance,setDistance]=useState('comfortable'),[speed,setSpeed]=useState('normal'),[orientation,setOrientation]=useState('front')
  const [scenario,setScenario]=useState<string>('known_word'),[observed,setObserved]=useState(false),[outcome,setOutcome]=useState('not_assessed')
  const [observations,setObservations]=useState<Record<string,unknown>[]>([])
  const current=useRef<StudioRecording|null>(null),started=useRef(0),alive=useRef(true),cropPreview=useRef<HTMLCanvasElement>(null),generation=useRef(0),abort=useRef<AbortController|null>(null)
  const phaseRef=useRef(phase);phaseRef.current=phase
  const live=cam.status==='live',busy=phase!=='ready',targets=config?.tasks[task].labels??[]
  useEffect(()=>{alive.current=true;void Promise.all([api('/config'),api('/inventory')]).then(([c,i])=>{if(alive.current){setConfig(c);setInventory(i)}}).catch(e=>setError(String(e)));return()=>{alive.current=false;generation.current++;abort.current?.abort();current.current=null}},[])
  const discard=()=>{current.current=null;setDraft(null);setConfirmed(false);setPlaying(false);setCount(0);setPhase('ready')}
  const stop=()=>{generation.current++;abort.current?.abort();cam.stop();discard();setUsable(false)}
  useEffect(()=>{if(cam.status==='denied'||cam.status==='error'){current.current=null;setDraft(null);setPhase('ready');setUsable(false)}},[cam.status])
  useEffect(()=>{
    if(!playing||!draft)return
    const i=reviewIndex+1
    const timer=setTimeout(()=>{if(i>=draft.frames.length){setPlaying(false);setReviewIndex(0)}else setReviewIndex(i)},Math.max(15,(draft.timestamps_ms[i]??draft.timestamps_ms[reviewIndex]+66)-draft.timestamps_ms[reviewIndex]))
    return()=>clearTimeout(timer)
  },[playing,draft,reviewIndex])
  useEffect(()=>{
    if(!live||lm.status!=='ready')return
    const canvas=document.createElement('canvas');canvas.width=canvas.height=96
    const ctx=canvas.getContext('2d',{willReadFrequently:true});let raf=0,last=0,videoTime=-1,sync=0
    const fail=(message:string)=>{current.current=null;setPhase('ready');setError(message);setCount(0)}
    const loop=(now:number)=>{
      raf=requestAnimationFrame(loop)
      const video=cam.videoRef.current,tracker=lm.landmarkerRef.current
      if(!video||!tracker||!ctx||video.readyState<2||video.currentTime===videoTime||now-last<66)return
      last=now;videoTime=video.currentTime
      try {
        const t0=performance.now(),faces=tracker.detectForVideo(video,now).faceLandmarks,t1=performance.now()
        const a=faces.length===1?mouthAlignment(faces[0],video.videoWidth,video.videoHeight):null
        let good=!!a,message=faces.length>1?'Keep exactly one face in view.':!a?'Center your face and keep the whole mouth visible.':'One face and usable mouth landmarks.'
        if(a){ctx.save();ctx.clearRect(0,0,96,96);ctx.translate(48,48);ctx.scale(96/a.size,96/a.size);ctx.rotate(-a.angle);ctx.drawImage(video,-a.cx,-a.cy);ctx.restore()
          const pixels=ctx.getImageData(0,0,96,96).data;let sum=0,sq=0
          for(let i=0;i<pixels.length;i+=4){const v=(pixels[i]+pixels[i+1]+pixels[i+2])/3;sum+=v;sq+=v*v}
          const mean=sum/9216,contrast=Math.sqrt(Math.max(0,sq/9216-mean*mean))
          if(mean<15||mean>240||contrast<8){good=false;message='Improve front lighting or reduce glare.'}
          cropPreview.current?.getContext('2d')?.drawImage(canvas,0,0)
        }
        const t2=performance.now(),row=current.current
        if(row){
          if(!good){fail('Face or mouth quality was lost; the incomplete capture was discarded.');return}
          if(now-started.current>5000||row.frames.length>=76||row.timestamps_ms.length&&now-started.current-row.timestamps_ms[row.timestamps_ms.length - 1]>250){fail('Capture exceeded its duration or frame-gap limit. Retry a complete utterance.');return}
          row.frames.push(canvas.toDataURL('image/jpeg',.75));row.timestamps_ms.push(now-started.current)
          row.quality.face_counts.push(faces.length);row.quality.mouth_valid.push(true)
          row.timing.detection_ms.push(t1-t0);row.timing.roi_ms.push(t2-t1);row.timing.jpeg_ms.push(performance.now()-t2)
        }
        if(now-sync>100){sync=now;setUsable(good);setFaceCount(faces.length);setQuality(message);if(row)setCount(row.frames.length)}
      }catch(e){setUsable(false);if(current.current)fail('Landmark processing failed; retry the camera.');else setError(String(e))}
    }
    raf=requestAnimationFrame(loop)
    const timer=setInterval(()=>{if(current.current&&performance.now()-started.current>5000)fail('Video stalled or capture timed out.')},150)
    return()=>{cancelAnimationFrame(raf);clearInterval(timer)}
  },[live,lm.status,cam.videoRef,lm.landmarkerRef])
  const start=async()=>{setError(null);if(await cam.start()==='live')await lm.load()}
  const begin=()=>{
    if(!config||busy||!usable||!consent)return
    try{checkSpeakerAssignment(pending,speaker,split);if(!/^[a-zA-Z0-9_-]{1,80}$/.test(speaker)||!/^[a-zA-Z0-9_-]{1,80}$/.test(session))throw new Error('Use valid participant/session IDs.')
      const video=cam.videoRef.current!
      current.current={id:crypto.randomUUID(),task,label,speaker_id:speaker,session_id:session,split,recorded_at:new Date().toISOString(),
        source:'user_camera',capture_kind:kind,preprocessing:config.preprocessing,landmarker_sha256:config.landmarker_sha256,
        frames:[],timestamps_ms:[],duration_ms:0,frame_count:0,frames_sha256:'',consent:true,label_confirmed:false,spoken_text:spoken,
        camera:{width:video.videoWidth,height:video.videoHeight,captured_fps:0},quality:{status:'passed',face_counts:[],mouth_valid:[]},
        conditions:{lighting,distance,speed,orientation},timing:{detection_ms:[],roi_ms:[],jpeg_ms:[]}}
      started.current=performance.now();setCount(0);setError(null);setNotice(null);setPhase('capturing')
    }catch(e){setError(String(e))}
  }
  const finish=async()=>{
    const row=current.current;current.current=null
    if(!row||row.frames.length<8){setError('Capture at least eight usable frames of the complete utterance.');setPhase('ready');return}
    row.frame_count=row.frames.length;row.duration_ms=row.timestamps_ms[row.timestamps_ms.length - 1]-row.timestamps_ms[0];row.camera.captured_fps=measuredFps(row.timestamps_ms)??0
    try{await verifyPlayback(row.frames);row.frames_sha256=await digest(new TextEncoder().encode(JSON.stringify(row.frames)));if(!alive.current)return
      setDraft(row);setReviewIndex(0);setConfirmed(false);setPhase('review')
    }catch(e){setError(String(e));setPhase('ready')}
  }
  const queue=()=>{
    if(!draft||!config||!confirmed)return
    try{const row={...draft,label_confirmed:true};checkStudioRecording(row,config);checkSpeakerAssignment(pending,row.speaker_id,row.split)
      if(pending.some(r=>r.id===row.id||r.frames_sha256===row.frames_sha256))throw new Error('This recording is already pending.')
      setPending(p=>[...p,row]);setNotice('Added to the pending bundle. Save it to validate decoded pixels and speaker assignments.');discard()
    }catch(e){setError(String(e))}
  }
  const save=async()=>{
    if(!config||!pending.length)return
    setPhase('saving');setError(null)
    try{await api('/bundles',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({clips:pending,configuration_sha256:config.configuration_sha256})})
      if(!alive.current)return;setPending([]);setInventory(await api('/inventory'));setNotice('Recordings validated and saved. No model was trained or activated.')
    }catch(e){setError(String(e))}finally{if(alive.current)setPhase('ready')}
  }
  const testClip=async()=>{
    if(!draft||!confirmed||task==='negative'&&!spoken&&label!=='silence')return
    const token=++generation.current,controller=new AbortController();abort.current=controller;setPhase('processing');setError(null)
    const began=performance.now()
    try{const endpoint=task==='alphabet_recognition'?'/api/predict':'/api/predict/word'
      const value=task==='alphabet_recognition'?await requestPrediction(draft.frames,controller.signal):await requestWordPrediction(draft.frames,controller.signal,'word')
      if(!alive.current||token!==generation.current)return
      const end=performance.now(),prediction='predicted_word'in value?value.word??null:value.uncertain?null:value.letter
      const row={id:crypto.randomUUID(),test_case:scenario,scope:kind==='physical_webcam'?'physical_webcam':'webcam_replay',
        speaker_id:speaker,session_id:session,task,expected_label:label,spoken_text:spoken,recording_id:draft.id,source_sha256:draft.frames_sha256,
        observer_confirmed:observed,assessment:outcome,actual_prediction:prediction,confidence:value.confidence,
        capture_ms:draft.duration_ms,api_round_trip_ms:end-began,inference_ms:value.latency_ms??null,
        confirmation_ms:null,total_capture_to_response_ms:draft.duration_ms+end-began,
        frontend_timing_scope:'Response state update; paint time not measured',endpoint,response:value,
        frame_timing:draft.timing,recorded_at:new Date().toISOString()}
      setObservations(o=>[...o,row]);setNotice(`Actual backend response: ${prediction??'UNCERTAIN'}. Assess the result, then export the session. Scores are uncalibrated.`)
    }catch(e){if(token===generation.current)setError(String(e))}finally{if(alive.current&&token===generation.current)setPhase('review')}
  }
  const observeCondition=()=>{
    if(!observed||!consent)return
    setObservations(o=>[...o,{id:crypto.randomUUID(),test_case:scenario,scope:kind,speaker_id:speaker,session_id:session,
      observer_confirmed:true,assessment:outcome,camera_status:cam.status,camera_error:cam.error??null,face_count:faceCount,mouth_usable:usable,
      quality,actual_prediction:null,confidence:null,capture_ms:null,inference_ms:null,confirmation_ms:null,total_capture_to_response_ms:null,
      failure_reason:cam.error??(!usable?quality:null),measurement_scope:'Observer condition record; no model inference claim',recorded_at:new Date().toISOString()}])
    setNotice('Condition observation added. It is separate from speech accuracy measurements.')
  }
  return <div className="word-collection"><header className="ws-page-header"><div><p className="ws-kicker">LABELLED DATA / SPEAKER INDEPENDENT</p><h1 className="ws-title">Recording <em>studio.</em></h1><p className="ws-lede">Collect complete utterances, review them, and preserve independent speakers. Saved data does not change an active model.</p></div><Link to="/demo/live">Open live recognition →</Link></header>
    {!config&&<p role="status">Loading the recording protocol…</p>}
    <div className="wc-fields"><label>Recording task<select aria-label="Recording task" disabled={busy} value={task} onChange={e=>{const t=e.target.value as StudioTask;setTask(t);setLabel(config?.tasks[t].labels[0]??'')}}><option value="isolated_word">Isolated words</option><option value="alphabet_recognition">Alphabet A–Z collection</option><option value="negative">Negative / OOV validation</option></select></label>
      <label>Target label<select aria-label="Target label" disabled={busy} value={label} onChange={e=>setLabel(e.target.value)}>{targets.map(w=><option key={w}>{w}</option>)}</select></label>
      <label>Participant ID<input aria-label="Participant ID" value={speaker} disabled={busy||live} onChange={e=>setSpeaker(e.target.value)}/></label>
      <label>Session ID<input value={session} disabled={busy||live} onChange={e=>setSession(e.target.value)}/></label>
      <label>Speaker partition<select disabled={busy||live} value={split} onChange={e=>setSplit(e.target.value as StudioSplit)}><option value="train">Training</option><option value="valid">Validation</option><option value="test">Final held-out test</option></select></label>
      <label>Input source<select disabled={busy||live} value={kind} onChange={e=>setKind(e.target.value as typeof kind)}><option value="physical_webcam">Physical webcam (observer declaration)</option><option value="webcam_replay">Genuine video replay / virtual input</option></select></label></div>
    <p>{config?.tasks[task].instruction}</p><p className="wc-hint">W has no GRID training examples. HELP, I and AM are collection targets with no trained eight-word output. No full A–Z or live accuracy is established.</p>
    <label className="wc-consent"><input type="checkbox" disabled={busy} checked={consent} onChange={e=>setConsent(e.target.checked)}/>I consent to storing this labelled mouth video for this project.</label>
    <div className="wc-layout"><section className="ws-card"><div className="wc-video"><video ref={cam.videoRef} muted playsInline aria-label="Collection camera preview"/>{!live&&<p>Camera off · no microphone</p>}</div>
      <div className="wc-actions">{!live?<button className="ws-btn ws-btn-primary" disabled={!consent||cam.status==='requesting'} onClick={()=>void start()}>Start camera</button>:<button className="ws-btn ws-btn-ghost" onClick={stop}>Stop camera</button>}<span role="status">{quality}</span></div>
      <p>Faces: {faceCount} · Mouth: {usable?'usable':'not usable'} · Tracker: {lm.status}</p><canvas ref={cropPreview} width={96} height={96} aria-label="Aligned mouth preview"/>
      {cam.error&&<p role="alert">{cam.error}</p>}{lm.error&&<p role="alert">{lm.error}</p>}
    </section><section className="ws-card wc-prompt"><span className="ws-kicker">SAY ONLY THE PROMPTED LABEL</span><h2>{label.toUpperCase()}</h2>
      <div className="wc-fields">{([['Lighting',lighting,setLighting,['normal','dim','bright']],['Distance',distance,setDistance,['comfortable','near','far']],['Speaking speed',speed,setSpeed,['normal','slow','fast']],['Orientation',orientation,setOrientation,['front','slightly_left','slightly_right']]] as const).map(([title,value,setter,choices])=><label key={title}>{title}<select value={value} disabled={busy} onChange={e=>setter(e.target.value)}>{choices.map(c=><option key={c}>{c}</option>)}</select></label>)}</div>
      {task==='negative'&&label==='unknown_word'&&<label>Actual isolated word spoken<input value={spoken} disabled={busy} onChange={e=>setSpoken(e.target.value.toLowerCase())}/></label>}
      <p aria-live="polite">{phase==='capturing'?'Capturing the complete utterance…':phase==='review'?'Play back and confirm the spoken label.':phase==='processing'?'Running the actual selected model…':phase==='saving'?'Validating and saving…':'Ready to record.'}</p><p>{count} / 76 frames</p>
      <div className="wc-actions">{phase==='ready'&&<button className="ws-btn ws-btn-primary" disabled={!live||!usable||!consent||!config} onClick={begin}>Record utterance</button>}{phase==='capturing'&&<button className="ws-btn ws-btn-primary" disabled={count<8} onClick={()=>void finish()}>Finish utterance</button>}</div>
      {draft&&<div className="wc-review-panel"><img src={draft.frames[reviewIndex]} width={192} height={192} alt="Recorded utterance playback"/><button className="ws-btn ws-btn-ghost" disabled={phase!=='review'} onClick={()=>{setReviewIndex(0);setPlaying(p=>!p)}}>{playing?'Stop playback':'Play complete clip'}</button><p>{draft.frame_count} frames · {draft.camera.captured_fps.toFixed(1)} captured FPS · {draft.camera.width}×{draft.camera.height} camera · {Math.round(draft.duration_ms)} ms</p>
        <label><input type="checkbox" disabled={phase!=='review'} checked={confirmed} onChange={e=>setConfirmed(e.target.checked)}/>I reviewed the complete clip and confirm that its label is “{draft.label}”.</label>
        <div className="wc-actions"><button className="ws-btn ws-btn-primary" disabled={!confirmed||phase!=='review'||pending.length>=32} onClick={queue}>Add to pending bundle</button><button className="ws-btn ws-btn-ghost" disabled={phase==='processing'} onClick={discard}>Discard and retry</button></div></div>}
    </section></div>
    {error&&<p className="wc-error" role="alert">{error}</p>}{notice&&<p className="wc-notice" role="status">{notice}</p>}
    <section className="ws-card"><h2>Pending data · {pending.length} clips</h2><p>Pending clips remain in memory until saved or exported. Existing server recordings cannot be overwritten.</p><div className="wc-actions"><button className="ws-btn ws-btn-primary" disabled={busy||!pending.length} onClick={()=>void save()}>Validate and save bundle</button><button className="ws-btn ws-btn-ghost" disabled={!pending.length} onClick={()=>downloadJson({format:'lipnet-studio-v2',configuration:config,recordings:pending},`pending-studio-${Date.now()}.json`)}>Export pending backup</button><button className="ws-btn ws-btn-ghost" disabled={busy} onClick={()=>void api(`/export/${task}`).then(v=>downloadJson(v,`${task}-${Date.now()}.json`)).catch(e=>setError(String(e)))}>Export saved task</button></div></section>
    <section className="ws-card"><h2>Class coverage and speaker diversity</h2><p>{config?.collection_recommendation}</p><p>{config?.final_test_rule}</p><p>Server: {inventory?.samples??0} validated recordings · {inventory?.unique_speakers??0} unique speakers. Historic corpora are reported separately in the inventory report.</p>
      <p>{inventory?.tasks[task].reason}</p><div className="wc-counts">{inventory?.tasks[task].labels.map(r=><button key={r.label} disabled={busy} aria-pressed={label===r.label} onClick={()=>setLabel(r.label)}><strong>{r.label}</strong><span>{r.samples} recordings · {r.speakers} speakers</span><small>train {r.splits.train} / valid {r.splits.valid} / test {r.splits.test}</small><small>Missing: {r.missing_splits.join(', ')||'review diversity'}</small><small>{r.quality_warnings} quality warnings</small></button>)}</div></section>
    <section className="ws-card"><h2>Physical webcam acceptance log</h2><p>Perform each case with real participants. Replay and software checks never count as physical speech evidence. Timing excludes unmeasured paint and physical speech onset.</p>
      <label>Test case<select value={scenario} onChange={e=>setScenario(e.target.value)}>{ACCEPTANCE_CASES.map(c=><option key={c}>{c}</option>)}</select></label><label>Observer assessment<select value={outcome} onChange={e=>setOutcome(e.target.value)}><option value="not_assessed">Not assessed</option><option value="correct">Correct</option><option value="incorrect">Incorrect</option><option value="rejected">Rejected</option><option value="condition_handled">Condition handled</option><option value="condition_failed">Condition failed</option></select></label>
      <label><input type="checkbox" checked={observed} onChange={e=>setObserved(e.target.checked)}/>I performed and observed this test case with the declared input source.</label>
      <div className="wc-actions"><button className="ws-btn ws-btn-primary" disabled={!draft||!confirmed||!observed||phase!=='review'} onClick={()=>void testClip()}>Test reviewed clip through real API</button><button className="ws-btn ws-btn-ghost" disabled={!observed||!consent||phase==='processing'} onClick={observeCondition}>Log observed camera/quality condition</button><button className="ws-btn ws-btn-ghost" disabled={!observations.length} onClick={()=>downloadJson({format:'lipnet-acceptance-v2',observations},`webcam-acceptance-${Date.now()}.json`)}>Export acceptance log ({observations.length})</button></div>
      <p>No live accuracy is calculated from an empty, unassessed or single-speaker session. Exported results preserve scope, actual API output and missing timings as null.</p>
    </section>
  </div>
}
