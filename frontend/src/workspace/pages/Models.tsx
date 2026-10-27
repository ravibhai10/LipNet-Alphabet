import { useCallback, useEffect, useMemo, useState } from "react"
import { inferenceCheckpoints, trainingCheckpoints, type InferenceCheckpoint, type TrainingCheckpoint } from "../api"
import { Icon } from "../Icon"

export function ModelsPage() {
  const [train, setTrain] = useState<TrainingCheckpoint[] | null>(null)
  const [infer, setInfer] = useState<InferenceCheckpoint[] | null>(null)
  const [modelLoaded, setModelLoaded] = useState<boolean | undefined>()
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState("all")
  const [selected, setSelected] = useState<TrainingCheckpoint | InferenceCheckpoint | null>(null)

  const refresh = useCallback(async () => {
    setLoading(true)
    const [t, i] = await Promise.all([trainingCheckpoints(), inferenceCheckpoints()])
    setTrain(t?.checkpoints ?? null)
    setInfer(i?.checkpoints ?? null)
    setModelLoaded(i?.model_loaded)
    setLoading(false)
  }, [])
  useEffect(() => { void refresh() }, [refresh])
  const match = (c: { name: string; path: string }) => `${c.name} ${c.path}`.toLowerCase().includes(query.trim().toLowerCase())
  const training = useMemo(() => (train ?? []).filter(match), [train, query])
  const inference = useMemo(() => (infer ?? []).filter(c => match(c) && (filter === "all" || c.compatible)), [infer, query, filter])

  return <div className="ws-page-inner">
    <header className="ws-page-header">
      <p className="ws-kicker">06 / Model registry</p>
      <h1 className="ws-title">The models behind<br /><em>every prediction.</em></h1>
      <p className="ws-lede">Inspect the checkpoints available to your local workspace, from training artifacts to inference compatibility.</p>
    </header>
    <div className="ws-grid-3">
      <div className="ws-card ws-metric"><span className="ws-metric-label">Training checkpoints</span><span className="ws-metric-value">{loading ? "…" : train?.length ?? "—"}</span><span className="ws-note">Available training artifacts</span></div>
      <div className="ws-card ws-metric"><span className="ws-metric-label">Inference checkpoints</span><span className="ws-metric-value">{loading ? "…" : infer?.length ?? "—"}</span><span className="ws-note">Discovered by the inference service</span></div>
      <div className="ws-card ws-metric"><span className="ws-metric-label">Model connection</span><span className="ws-metric-value" style={{ fontSize: "28px" }}>{loading ? "Checking" : modelLoaded === undefined ? "Unknown" : modelLoaded ? "Loaded" : "Not loaded"}</span><span className="ws-note">Reported by the backend</span></div>
    </div>
    {!loading && (train === null || infer === null) && <div className="ws-error" role="alert">{train === null && infer === null ? "The model registry is unavailable. Check the backend connection and retry." : "Some registry data could not be loaded. Unavailable sources are marked below."}</div>}
    <div className="ws-toolbar">
      <label className="ws-field"><span>Search checkpoints</span><input className="ws-input" type="search" placeholder="Name or path…" value={query} onChange={e => setQuery(e.target.value)} /></label>
      <label className="ws-field"><span>Inference compatibility</span><select className="ws-select ws-input" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All checkpoints</option><option value="compatible">Compatible only</option></select></label>
      <button className="ws-btn ws-btn-ghost" onClick={() => void refresh()} disabled={loading}><Icon name="refresh" />{loading ? "Refreshing…" : "Refresh"}</button>
    </div>
    <section className="ws-card">
      <div className="ws-card-head"><div><p className="ws-kicker">Available for inference</p><h2>Inference registry</h2></div><span className="ws-badge ws-badge-muted">{inference.length} shown</span></div>
      {loading ? <p className="ws-note" role="status">Reading inference metadata…</p> : infer === null ? <p className="ws-note">Inference registry unavailable.</p> : inference.length === 0 ? <div className="ws-empty"><Icon name="layers" size={30} /><h3>{query || filter !== "all" ? "No matching checkpoints" : "No inference checkpoints found"}</h3><p>{query || filter !== "all" ? "Try a different search or compatibility filter." : "A compatible trained checkpoint will appear here when the backend discovers it."}</p></div> : <div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Checkpoint</th><th>Compatibility</th><th>Details</th><th><span className="sr-only">Inspect</span></th></tr></thead><tbody>{inference.map(c => <tr key={c.path}><td><strong>{c.name}</strong><small className="ws-code">{c.path}</small></td><td><span className={`ws-badge ${c.compatible ? "ws-badge-ready" : "ws-badge-warning"}`}>{c.compatible ? "Compatible" : "Incompatible"}</span></td><td>{c.note || "No additional metadata"}</td><td><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setSelected(c)} aria-label={`Inspect ${c.name}`}><Icon name="arrow" /></button></td></tr>)}</tbody></table></div>}
      <p className="ws-note">The inference service chooses the active model. Switching and downloading checkpoints are not available in this workspace.</p>
    </section>
    <section className="ws-card">
      <div className="ws-card-head"><div><p className="ws-kicker">Training outputs</p><h2>Checkpoint archive</h2></div><Icon name="database" /></div>
      {loading ? <p className="ws-note">Reading checkpoint metadata…</p> : train === null ? <p className="ws-note">Training archive unavailable.</p> : training.length === 0 ? <div className="ws-empty"><Icon name="database" size={30} /><h3>{query ? "No matching outputs" : "Your next model starts here"}</h3><p>{query ? "Try another checkpoint name." : "Complete a training run to populate the checkpoint archive."}</p></div> : <div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Checkpoint</th><th>Created</th><th>Epoch</th><th>Best validation metric</th><th><span className="sr-only">Inspect</span></th></tr></thead><tbody>{training.map(c => <tr key={c.path}><td><strong>{c.name}</strong></td><td className="ws-mono">{c.created || "Not provided"}</td><td>{c.epoch ?? "—"}</td><td>{c.best_val != null ? c.best_val.toFixed(4) : "Not provided"}</td><td><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setSelected(c)} aria-label={`Inspect ${c.name}`}><Icon name="arrow" /></button></td></tr>)}</tbody></table></div>}
    </section>
    {selected && <section className="ws-card" aria-label="Checkpoint details">
      <div className="ws-card-head"><div><p className="ws-kicker">Checkpoint details</p><h2>{selected.name}</h2></div><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setSelected(null)} aria-label="Close checkpoint details"><Icon name="close" /></button></div>
      <dl className="ws-detail-list"><div><dt>Storage path</dt><dd className="ws-code">{selected.path}</dd></div>{"compatible" in selected ? <div><dt>Compatibility</dt><dd>{selected.compatible ? "Compatible with inference service" : "Not compatible"}</dd></div> : <><div><dt>Configuration hash</dt><dd className="ws-code">{selected.config_hash || "Not provided"}</dd></div><div><dt>Representation version</dt><dd>{selected.repr_version || "Not provided"}</dd></div><div><dt>Vocabulary</dt><dd>{selected.vocab || "Not provided"}</dd></div></>}</dl>
    </section>}
  </div>
}
