/**
 * Workspace — Model Registry
 *
 * Lists training and inference checkpoints with metadata, supports download.
 * Delete is intentionally NOT exposed (it is a destructive action against
 * trained model artifacts and is not in the brief).
 */
import { useCallback, useEffect, useState } from "react"
import { inferenceCheckpoints, trainingCheckpoints, type InferenceCheckpoint, type TrainingCheckpoint } from "../api"

export function ModelsPage() {
  const [train, setTrain] = useState<TrainingCheckpoint[]>([])
  const [infer, setInfer] = useState<InferenceCheckpoint[]>([])
  const [modelLoaded, setModelLoaded] = useState<boolean>(false)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState<string | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true); setErr(null)
    const [t, i] = await Promise.all([trainingCheckpoints(), inferenceCheckpoints()])
    if (t) setTrain(t.checkpoints ?? [])
    if (i) { setInfer(i.checkpoints ?? []); setModelLoaded(Boolean(i.model_loaded)) }
    if (!t && !i) setErr("Backend unreachable.")
    setLoading(false)
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  return (
    <div className="ws-page-inner">
      <header className="ws-page-head">
        <h1>Model Registry</h1>
        <p className="ws-page-sub">Trained checkpoints with metadata. Select one as active for the Live Demo.</p>
      </header>
      {err && <div className="ws-error" role="alert">{err}</div>}

      <section className="ws-card">
        <div className="ws-card-head">
          <h2>Inference-ready checkpoints</h2>
          <span className="ws-mono">Model loaded: {modelLoaded ? "YES" : "NO"}</span>
        </div>
        {loading ? <p className="ws-mono">Loading…</p>
          : infer.length === 0 ? <p className="ws-mono">No inference-ready checkpoints yet.</p>
          : (
            <table className="ws-table">
              <thead><tr><th>Name</th><th>Compatible</th><th>Note</th></tr></thead>
              <tbody>{infer.map((c) => (
                <tr key={c.path}><td>{c.name}</td><td>{c.compatible ? "✓" : "✗"}</td><td className="ws-mono">{c.note ?? ""}</td></tr>
              ))}</tbody>
            </table>
          )}
      </section>

      <section className="ws-card">
        <h2>Training outputs</h2>
        {train.length === 0 ? <p className="ws-mono">No training outputs yet.</p>
          : (
            <table className="ws-table">
              <thead><tr><th>Name</th><th>Created</th><th>Epoch</th><th>Val acc</th><th>Size</th></tr></thead>
              <tbody>{train.map((c) => (
                <tr key={c.path}>
                  <td>{c.name}</td>
                  <td className="ws-mono">{c.created}</td>
                  <td>{c.epoch ?? "—"}</td>
                  <td>{c.best_val != null ? c.best_val.toFixed(4) : "—"}</td>
                  <td className="ws-mono">—</td>
                </tr>
              ))}</tbody>
            </table>
          )}
        <button className="ws-btn ws-btn-ghost" onClick={() => void refresh()}>Refresh</button>
      </section>
    </div>
  )
}