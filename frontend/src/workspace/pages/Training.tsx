import { useCallback, useEffect, useRef, useState, type FormEvent } from "react"
import { Link } from "react-router-dom"
import { cancelTraining, getWorkspaceError, startTraining, trainingCheckpoints, trainingPrecheck, trainingStatus, type TrainingCheckpoint, type TrainingJobStatus, type TrainingPrecheck } from "../api"
import { useBackend } from "../../hooks/useBackend"
import { Icon } from "../Icon"

const JOB_KEY = "lipnet.workspace.training-job"
function savedJob() { try { return sessionStorage.getItem(JOB_KEY) } catch { return null } }

export function TrainingPage() {
  const { refresh } = useBackend()
  const [precheck, setPrecheck] = useState<TrainingPrecheck | null>(null)
  const [precheckLoading, setPrecheckLoading] = useState(true)
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpoint[] | null>(null)
  const [cpLoading, setCpLoading] = useState(true)
  const [jobId, setJobId] = useState<string | null>(savedJob)
  const [jobStatus, setJobStatus] = useState<TrainingJobStatus | null>(null)
  const [epochs, setEpochs] = useState("30")
  const [batchSize, setBatchSize] = useState("16")
  const [lr, setLr] = useState("0.001")
  const [submitting, setSubmitting] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [actionError, setActionError] = useState("")
  const [pollError, setPollError] = useState("")
  const [cancelNote, setCancelNote] = useState("")
  const startGuard = useRef(false)
  const refreshRef = useRef(refresh)
  refreshRef.current = refresh
  const loadPrecheck = useCallback(async () => { setPrecheckLoading(true); setPrecheck(await trainingPrecheck()); setPrecheckLoading(false) }, [])
  const loadCheckpoints = useCallback(async () => { setCpLoading(true); const result = await trainingCheckpoints(); setCheckpoints(result?.checkpoints ?? null); setCpLoading(false) }, [])
  useEffect(() => { void loadPrecheck(); void loadCheckpoints() }, [loadPrecheck, loadCheckpoints])
  useEffect(() => {
    if (!jobId) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => {
      const result = await trainingStatus(jobId)
      if (!active) return
      if (result) {
        setJobStatus(result); setPollError("")
        if (["COMPLETED", "FAILED", "CANCELLED"].includes(result.state)) {
          if (result.state === "COMPLETED") { void loadCheckpoints(); refreshRef.current() }
          return
        }
      } else setPollError("Job status could not be refreshed. Retrying automatically; the job state is unconfirmed.")
      timer = setTimeout(() => void poll(), 3000)
    }
    void poll()
    return () => { active = false; clearTimeout(timer) }
  }, [jobId, loadCheckpoints])
  const running = !!jobId && (!jobStatus || jobStatus.state === "RUNNING" || jobStatus.state === "QUEUED" || jobStatus.state === "UNKNOWN")
  const progress = jobStatus?.total_epochs && jobStatus.epoch !== undefined ? Math.min(100, Math.max(0, jobStatus.epoch / jobStatus.total_epochs * 100)) : null

  async function begin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (startGuard.current || running) return
    const config = { epochs: Number(epochs), batch_size: Number(batchSize), lr: Number(lr) }
    if (!Number.isInteger(config.epochs) || config.epochs < 1 || config.epochs > 500 || !Number.isInteger(config.batch_size) || config.batch_size < 1 || config.batch_size > 256 || !Number.isFinite(config.lr) || config.lr < 0.000001 || config.lr > 1) {
      setActionError("Use 1–500 epochs, a batch size of 1–256, and a learning rate between 0.000001 and 1."); return
    }
    startGuard.current = true; setSubmitting(true); setActionError(""); setCancelNote("")
    try {
      const result = await startTraining(config)
      if (!result?.job_id) { setActionError(getWorkspaceError() || "Training could not start. Recheck dataset readiness and the backend connection."); return }
      setJobId(result.job_id); setJobStatus(null)
      try { sessionStorage.setItem(JOB_KEY, result.job_id) } catch { /* Polling still works without session storage. */ }
    } finally { setSubmitting(false); startGuard.current = false }
  }
  async function cancel() {
    if (!jobId || cancelling) return
    setCancelling(true); setActionError("")
    try {
      const result = await cancelTraining(jobId)
      if (!result?.ok) { setActionError(result?.note || getWorkspaceError() || "Cancellation was not confirmed. The job may still be running."); return }
      setCancelNote("Cancellation request acknowledged. Waiting for the backend to confirm the job state.")
      const actual = await trainingStatus(jobId)
      if (actual) setJobStatus(actual)
    } finally { setCancelling(false) }
  }
  function dismissJob() { setJobId(null); setJobStatus(null); setPollError(""); setCancelNote(""); try { sessionStorage.removeItem(JOB_KEY) } catch { /* Optional local persistence. */ } }

  return <div className="ws-page-inner">
    <header className="ws-page-header"><p className="ws-kicker">03 / Training lab</p><h1 className="ws-title">Turn your recordings<br /><em>into understanding.</em></h1><p className="ws-lede">Verify your dataset, configure a training run, and follow the progress reported by your local backend.</p><p className="ws-note">This form trains the alphabet CNN-BiLSTM. The AV-HuBERT head workflow uses a separate labeled feature cache and speaker-disjoint evaluation.</p></header>
    <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Before you begin</p><h2>Dataset readiness</h2></div><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={precheckLoading} onClick={() => void loadPrecheck()}><Icon name="refresh" />{precheckLoading ? "Checking…" : "Recheck"}</button></div>
      <div className="ws-grid-3"><div className="ws-metric"><span className="ws-metric-label">Training gate</span><span className="ws-metric-value" style={{fontSize:"28px"}}>{precheckLoading ? "Checking" : precheck ? precheck.can_start ? "Ready" : "Needs attention" : "Unavailable"}</span></div><div className="ws-metric"><span className="ws-metric-label">Recordings</span><span className="ws-metric-value">{precheckLoading ? "…" : precheck?.dataset_summary?.recordings?.toLocaleString() ?? "—"}</span></div><div className="ws-metric"><span className="ws-metric-label">Speakers</span><span className="ws-metric-value">{precheckLoading ? "…" : precheck?.dataset_summary?.speakers ?? "—"}</span></div></div>
      {!precheckLoading && !precheck && <p className="ws-error" role="alert">The training precheck is unavailable. Check the backend connection and retry.</p>}
      {!!precheck?.reasons.length && <div className="ws-error" role="status"><p>Resolve these checks before starting:</p><ul>{precheck.reasons.map((reason, i) => <li key={i}>{reason}</li>)}</ul><Link className="ws-btn ws-btn-ghost ws-btn-sm" to="/demo/dataset">Open dataset explorer <Icon name="arrow" /></Link></div>}
      {!!precheck?.warnings?.length && <details className="ws-details"><summary>{precheck.warnings.length} dataset notes</summary><ul>{precheck.warnings.map((warning, i) => <li key={i}>{warning}</li>)}</ul></details>}
    </section>
    <div className="ws-grid-2"><section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Training configuration</p><h2>A considered starting point</h2></div><span className="ws-badge ws-badge-muted"><Icon name="cpu" size={13} />Local CPU</span></div><form onSubmit={event => void begin(event)}><div className="ws-grid-3"><label className="ws-field"><span>Epochs</span><input className="ws-input" type="number" value={epochs} min={1} max={500} step={1} required disabled={running || submitting} onChange={e => setEpochs(e.target.value)} /><small>1–500 passes</small></label><label className="ws-field"><span>Batch size</span><input className="ws-input" type="number" value={batchSize} min={1} max={256} step={1} required disabled={running || submitting} onChange={e => setBatchSize(e.target.value)} /><small>Samples per step</small></label><label className="ws-field"><span>Learning rate</span><input className="ws-input" type="number" value={lr} min={0.000001} max={1} step="any" required disabled={running || submitting} onChange={e => setLr(e.target.value)} /><small>Optimizer step size</small></label></div><p className="ws-note">Training runs on the local backend. Smaller batches can help limit memory use on a CPU laptop.</p><div className="ws-action-row"><button className="ws-btn ws-btn-primary" type="submit" disabled={submitting || running || precheckLoading || !precheck?.can_start}><Icon name="play" />{submitting ? "Submitting…" : running ? "Training job active" : "Start training"}</button>{running && <button type="button" className="ws-btn ws-btn-ghost" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? "Requesting…" : "Request cancellation"}</button>}</div></form>{actionError && <p className="ws-error" role="alert">{actionError}</p>}{cancelNote && <p className="ws-note" role="status">{cancelNote}</p>}</section>
      <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Job telemetry</p><h2>Training progress</h2></div>{jobStatus && <span className={`ws-badge ${jobStatus.state === "COMPLETED" ? "ws-badge-ready" : jobStatus.state === "FAILED" ? "ws-badge-warning" : "ws-badge-muted"}`}>{jobStatus.state.toLowerCase()}</span>}</div>
        {!jobId ? <div className="ws-empty"><Icon name="activity" size={34} /><h3>Ready for the next iteration</h3><p>Start a training run to see the job state and any metrics reported by the backend.</p></div> : <><p className="ws-mono">Job {jobId}</p>{!jobStatus && <p className="ws-note" role="status">Waiting for the first job status…</p>}{progress !== null ? <div className="ws-progress-bar" role="progressbar" aria-label="Training epoch progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(progress)}><div className="ws-progress-fill" style={{width:`${progress}%`}} /></div> : <p className="ws-note">A progress percentage is not currently reported.</p>}<dl className="ws-detail-list"><div><dt>Epoch</dt><dd>{jobStatus?.epoch ?? "Not reported"}{jobStatus?.total_epochs ? ` / ${jobStatus.total_epochs}` : ""}</dd></div><div><dt>Training loss</dt><dd>{jobStatus?.train_loss?.toFixed(4) ?? "Not reported"}</dd></div><div><dt>Validation loss</dt><dd>{jobStatus?.val_loss?.toFixed(4) ?? "Not reported"}</dd></div></dl>{jobStatus?.message && <p className={jobStatus.state === "FAILED" ? "ws-error" : "ws-note"}>{jobStatus.message}</p>}{jobStatus && !running && <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={dismissJob}>Dismiss completed job</button>}</>}
        {pollError && <p className="ws-error" role="alert">{pollError}</p>}
      </section></div>
    <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Saved outputs</p><h2>Training checkpoints</h2></div><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={cpLoading} onClick={() => void loadCheckpoints()}><Icon name="refresh" />Refresh</button></div>{cpLoading ? <p className="ws-note" role="status">Reading saved checkpoints…</p> : checkpoints === null ? <p className="ws-note">Checkpoint metadata is unavailable.</p> : checkpoints.length === 0 ? <div className="ws-empty"><Icon name="layers" size={30} /><h3>No checkpoints yet</h3><p>Completed training outputs will appear here when saved by the backend.</p></div> : <div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Checkpoint</th><th>Created</th><th>Epoch</th><th>Best validation metric</th></tr></thead><tbody>{checkpoints.map(cp => <tr key={cp.path}><td><strong>{cp.name}</strong></td><td>{cp.created || "Not provided"}</td><td>{cp.epoch ?? "—"}</td><td>{cp.best_val?.toFixed(4) ?? "Not provided"}</td></tr>)}</tbody></table></div>}</section>
  </div>
}
