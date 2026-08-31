/**
 * Workspace — Training Lab
 *
 * Pre-training check (gate + manifest + splits), start training with
 * hyperparameters, live progress polling, checkpoint listing.
 */
import { useCallback, useEffect, useRef, useState } from "react"
import {
  trainingCheckpoints, trainingPrecheck, startTraining,
  trainingStatus, cancelTraining,
  type TrainingCheckpoint, type TrainingJobStatus, type TrainingPrecheck,
} from "../api"
import { useBackend } from "../../hooks/useBackend"

export function TrainingPage() {
  const { refresh } = useBackend()
  const [precheck, setPrecheck] = useState<TrainingPrecheck | null>(null)
  const [precheckLoading, setPrecheckLoading] = useState(true)
  const [jobId, setJobId] = useState<string | null>(null)
  const [jobStatus, setJobStatus] = useState<TrainingJobStatus | null>(null)
  const [startError, setStartError] = useState<string | null>(null)
  const [epochs, setEpochs] = useState(30)
  const [batchSize, setBatchSize] = useState(16)
  const [lr, setLr] = useState(0.001)
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpoint[]>([])
  const [cpLoading, setCpLoading] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = useCallback(() => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }, [])

  const loadPrecheck = useCallback(async () => {
    setPrecheckLoading(true)
    try { const r = await trainingPrecheck(); setPrecheck(r) }
    finally { setPrecheckLoading(false) }
  }, [])

  const loadCheckpoints = useCallback(async () => {
    setCpLoading(true)
    try { const r = await trainingCheckpoints(); if (r) setCheckpoints(r.checkpoints ?? []) }
    finally { setCpLoading(false) }
  }, [])

  useEffect(() => { void loadPrecheck(); void loadCheckpoints() }, [loadPrecheck, loadCheckpoints])

  useEffect(() => {
    if (!jobId) return
    const poll = async () => {
      const s = await trainingStatus(jobId)
      if (s) {
        setJobStatus(s)
        if (s.state === "COMPLETED" || s.state === "FAILED" || s.state === "CANCELLED") {
          stopPoll()
          if (s.state === "COMPLETED") { void loadCheckpoints(); void refresh() }
        }
      }
    }
    pollRef.current = setInterval(poll, 3000)
    return () => stopPoll()
  }, [jobId, stopPoll, loadCheckpoints, refresh])

  const handleStart = useCallback(async () => {
    setStartError(null)
    const result = await startTraining({ epochs, batch_size: batchSize, lr })
    if (!result) { setStartError("Failed to start — backend unreachable or job already running."); return }
    setJobId(result.job_id)
    setJobStatus({ job_id: result.job_id, state: "QUEUED" })
  }, [epochs, batchSize, lr])

  const handleCancel = useCallback(async () => {
    if (!jobId) return
    await cancelTraining(jobId)
    stopPoll()
    setJobStatus((s) => s ? { ...s, state: "CANCELLED" } : s)
  }, [jobId, stopPoll])

  const progress = jobStatus?.total_epochs ? Math.round((100 * (jobStatus.epoch ?? 0)) / jobStatus.total_epochs) : 0
  const isRunning = jobStatus?.state === "RUNNING" || jobStatus?.state === "QUEUED"

  return (
    <div className="ws-page-inner">
      <header className="ws-page-head">
        <h1>Training Lab</h1>
        <p className="ws-page-sub">Real-camera dataset → MediaPipe → Pre-training gate → Conformer training → CTC decode → A–Z</p>
      </header>

      <section className="ws-card">
        <h2>Pre-training check</h2>
        {precheckLoading ? <p className="ws-mono">Checking…</p>
          : precheck ? (
            <div>
              <div className="ws-grid-4">
                <div className="ws-metric">
                  <span className="ws-metric-label">Can start</span>
                  <span className={`ws-metric-value ${precheck.can_start ? "ws-ok" : "ws-fail"}`}>{precheck.can_start ? "YES" : "NO"}</span>
                </div>
                <div className="ws-metric">
                  <span className="ws-metric-label">Recordings</span>
                  <span className="ws-metric-value">{precheck.dataset_summary?.recordings ?? "—"}</span>
                </div>
                <div className="ws-metric">
                  <span className="ws-metric-label">Speakers</span>
                  <span className="ws-metric-value">{precheck.dataset_summary?.speakers ?? "—"}</span>
                </div>
                <div className="ws-metric">
                  <span className="ws-metric-label">Letters</span>
                  <span className="ws-metric-value">{precheck.dataset_summary?.letters != null ? `${precheck.dataset_summary.letters}/26` : "—"}</span>
                </div>
              </div>
              {precheck.reasons.length > 0 && (
                <ul className="ws-error-list">{precheck.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
              )}
              <button className="ws-btn ws-btn-ghost" onClick={() => void loadPrecheck()}>Re-check</button>
            </div>
          ) : <p className="ws-mono ws-error">Backend unreachable.</p>}
      </section>

      <section className="ws-card">
        <h2>Hyperparameters</h2>
        <div className="ws-grid-3">
          <label className="ws-field"><span>Epochs</span>
            <input type="number" className="ws-input" value={epochs} min={1} max={500} onChange={(e) => setEpochs(Number(e.target.value))} disabled={isRunning} />
          </label>
          <label className="ws-field"><span>Batch size</span>
            <input type="number" className="ws-input" value={batchSize} min={1} max={256} onChange={(e) => setBatchSize(Number(e.target.value))} disabled={isRunning} />
          </label>
          <label className="ws-field"><span>Learning rate</span>
            <input type="number" className="ws-input" value={lr} min={0.00001} max={1} step={0.0001} onChange={(e) => setLr(Number(e.target.value))} disabled={isRunning} />
          </label>
        </div>
        <div className="ws-action-row">
          <button className="ws-btn ws-btn-solid" onClick={() => void handleStart()} disabled={isRunning || !precheck?.can_start}>
            {isRunning ? "Training…" : "Start training"}
          </button>
          {isRunning && <button className="ws-btn ws-btn-ghost" onClick={() => void handleCancel()}>Cancel</button>}
        </div>
        {startError && <p className="ws-mono ws-error">{startError}</p>}
      </section>

      {jobStatus && (
        <section className="ws-card">
          <h2>Training progress</h2>
          <div className="ws-job-status-line">
            <span className={`ws-status-dot ws-status-${jobStatus.state.toLowerCase()}`} />
            <span className="ws-mono">{jobStatus.state}</span>
            {jobStatus.epoch != null && <span className="ws-mono">Epoch {jobStatus.epoch}{jobStatus.total_epochs ? ` / ${jobStatus.total_epochs}` : ""}</span>}
          </div>
          <div className="ws-progress-bar"><div className="ws-progress-fill" style={{ width: `${progress}%` }} /></div>
          <div className="ws-loss-row"><span className="ws-loss-label">Train loss</span><span className="ws-loss-value">{jobStatus.train_loss != null ? jobStatus.train_loss.toFixed(4) : "—"}</span></div>
          <div className="ws-loss-row"><span className="ws-loss-label">Val loss</span><span className="ws-loss-value">{jobStatus.val_loss != null ? jobStatus.val_loss.toFixed(4) : "—"}</span></div>
          <div className="ws-loss-row"><span className="ws-loss-label">Best val</span><span className="ws-loss-value">{jobStatus.best_val != null ? jobStatus.best_val.toFixed(4) : "—"}</span></div>
          {jobStatus.message && <p className="ws-mono">{jobStatus.message}</p>}
        </section>
      )}

      <section className="ws-card">
        <h2>Checkpoints</h2>
        {cpLoading ? <p className="ws-mono">Loading…</p>
          : checkpoints.length === 0 ? (
            <div className="ws-empty">
              <p className="ws-empty-icon">📦</p>
              <p>No trained checkpoints yet. Record data and train a model.</p>
            </div>
          ) : (
            <table className="ws-table">
              <thead><tr><th>Name</th><th>Created</th><th>Epoch</th><th>Val accuracy</th></tr></thead>
              <tbody>{checkpoints.map((cp) => (
                <tr key={cp.path}><td>{cp.name}</td><td>{cp.created}</td><td>{cp.epoch ?? "—"}</td><td>{cp.best_val != null ? cp.best_val.toFixed(4) : "—"}</td></tr>
              ))}</tbody>
            </table>
          )}
        <button className="ws-btn ws-btn-ghost" onClick={() => void loadCheckpoints()}>Refresh</button>
      </section>
    </div>
  )
}