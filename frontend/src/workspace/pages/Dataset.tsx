import { useCallback, useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { buildManifest, createSplits, datasetOverview, datasetRecordings, getWorkspaceError, validateDataset, type DatasetOverview, type DatasetRecording, type DatasetValidation } from "../api"
import { API_BASE } from "../../lib/api"
import { Icon } from "../Icon"

export function DatasetPage() {
  const [overview, setOverview] = useState<DatasetOverview | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState("")
  const [filter, setFilter] = useState("all")
  const [letter, setLetter] = useState("A")
  const [recordings, setRecordings] = useState<DatasetRecording[] | null>(null)
  const [recordingsLoading, setRecordingsLoading] = useState(true)
  const [selected, setSelected] = useState<DatasetRecording | null>(null)
  const [videoError, setVideoError] = useState(false)
  const [busy, setBusy] = useState<"validate" | "manifest" | "splits" | null>(null)
  const [validation, setValidation] = useState<DatasetValidation | null>(null)
  const [manifestEntries, setManifestEntries] = useState<number | null>(null)
  const [splits, setSplits] = useState<Record<string, number> | null>(null)
  const [message, setMessage] = useState("")
  const [error, setError] = useState("")
  const load = useCallback(async () => { setLoading(true); setOverview(await datasetOverview()); setLoading(false) }, [])
  useEffect(() => { void load() }, [load])
  useEffect(() => {
    let active = true
    setRecordingsLoading(true); setSelected(null); setVideoError(false)
    void datasetRecordings(letter).then(result => {
      if (!active) return
      setRecordings(result?.recordings ?? null); setRecordingsLoading(false)
    })
    return () => { active = false }
  }, [letter])
  async function prepare(action: "validate" | "manifest" | "splits") {
    if (busy) return
    setBusy(action); setMessage(""); setError("")
    try {
      if (action === "validate") {
        setManifestEntries(null); setSplits(null)
        const result = await validateDataset(); setValidation(result)
        if (!result) setError(getWorkspaceError() || "Validation is unavailable. Retry when the backend is connected.")
        else setMessage(result.ok ? "Validation passed. You can now build the manifest." : "Validation found issues. Review the checks before continuing.")
      } else if (action === "manifest") {
        setSplits(null)
        const result = await buildManifest()
        if (!result?.ok) setError(result?.note || getWorkspaceError() || "The manifest could not be built.")
        else { setManifestEntries(result.entries ?? null); setMessage(result.note || "Manifest built.") }
      } else {
        const result = await createSplits()
        if (!result?.ok) setError(result?.note || getWorkspaceError() || "Dataset splits could not be created.")
        else { setSplits(result.counts ?? {}); setMessage("Dataset splits saved. Run the training precheck to verify readiness.") }
      }
    } finally { setBusy(null) }
  }
  const letters = (overview?.letters ?? []).filter(item => item.letter.toLowerCase().includes(query.trim().toLowerCase()) && (filter === "all" || (filter === "recorded" ? item.total_recordings > 0 : item.total_recordings === 0)))

  return <div className="ws-page-inner">
    <header className="ws-page-header"><p className="ws-kicker">02 / Dataset explorer</p><h1 className="ws-title">Better data.<br /><em>Clearer understanding.</em></h1><p className="ws-lede">Explore your alphabet recordings, inspect real samples, and prepare the dataset for its next training run.</p></header>
    <div className="ws-grid-3">
      {[{ label: "Recordings", value: overview?.total_recordings, note: "Samples in your local dataset" }, { label: "Speakers", value: overview?.total_speakers, note: "Distinct recorded speakers" }, { label: "Sessions", value: overview?.total_sessions, note: "Distinct session identifiers" }].map(item => <div key={item.label} className="ws-card ws-metric"><span className="ws-metric-label">{item.label}</span><span className="ws-metric-value">{loading ? "…" : item.value?.toLocaleString() ?? "—"}</span><span className="ws-note">{item.note}</span></div>)}
    </div>
    <section className="ws-card">
      <div className="ws-card-head"><div><p className="ws-kicker">A — Z collection</p><h2>Explore the alphabet</h2></div><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={loading} onClick={() => void load()}><Icon name="refresh" />{loading ? "Refreshing…" : "Refresh"}</button></div>
      <div className="ws-toolbar"><label className="ws-field"><span>Search letters</span><input className="ws-input" type="search" placeholder="Find a letter…" value={query} maxLength={26} onChange={e => setQuery(e.target.value)} /></label><label className="ws-field"><span>Availability</span><select className="ws-input ws-select" value={filter} onChange={e => setFilter(e.target.value)}><option value="all">All letters</option><option value="recorded">With recordings</option><option value="empty">Without recordings</option></select></label></div>
      {loading ? <p className="ws-note" role="status">Reading dataset metadata…</p> : !overview ? <div className="ws-empty"><Icon name="database" size={32} /><h3>Dataset metadata unavailable</h3><p>Connect the backend to explore your local recordings.</p></div> : letters.length === 0 ? <div className="ws-empty"><h3>No letters match this filter</h3><p>Clear the search or choose another availability filter.</p></div> : <div className="ws-letter-grid">{letters.map(item => <button key={item.letter} className={`ws-letter-tile ${item.letter === letter ? "selected" : ""}`} onClick={() => setLetter(item.letter)} aria-pressed={item.letter === letter} aria-label={`${item.letter}, ${item.total_recordings} recordings`}><strong>{item.letter}</strong><span>{item.total_recordings} samples</span></button>)}</div>}
    </section>
    <section className="ws-card">
      <div className="ws-card-head"><div><p className="ws-kicker">Selected letter / {letter}</p><h2>Recording library</h2></div><span className="ws-badge ws-badge-muted">{recordingsLoading ? "Loading" : recordings === null ? "Unavailable" : `${recordings.length} samples`}</span></div>
      {recordingsLoading ? <p className="ws-note" role="status">Reading recordings for {letter}…</p> : recordings === null ? <p className="ws-error" role="alert">Recordings could not be loaded. Choose another letter or reload this page to retry.</p> : recordings.length === 0 ? <div className="ws-empty"><Icon name="camera" size={30} /><h3>No recordings for {letter} yet</h3><p>Recorded samples will appear here when they are added to the local dataset.</p></div> : <div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Recording</th><th>Speaker</th><th>Session</th><th>Duration</th><th>Preview</th></tr></thead><tbody>{recordings.map(item => <tr key={item.path}><td className="ws-code">{item.recording_id}</td><td>{item.speaker_id}</td><td>{item.session_id}</td><td>{item.duration_ms > 0 ? `${(item.duration_ms / 1000).toFixed(1)}s` : "Not provided"}</td><td><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => { setSelected(item); setVideoError(false) }} aria-label={`Preview recording ${item.recording_id}`}><Icon name="play" />View</button></td></tr>)}</tbody></table></div>}
      {selected && <div className="ws-recording-preview"><div className="ws-card-head"><h3>{selected.recording_id}</h3><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setSelected(null)} aria-label="Close recording preview"><Icon name="close" /></button></div>{videoError ? <p className="ws-error" role="alert">This recording could not be played. The video may be unavailable or in an unsupported format.</p> : <video key={selected.path} controls playsInline preload="metadata" aria-label={`Recorded sample ${letter}`} src={`${API_BASE}/api/video/${selected.path.split("/").map(encodeURIComponent).join("/")}`} onError={() => setVideoError(true)} style={{ width: "100%", maxHeight: "360px", borderRadius: "12px", background: "var(--ws-bg)" }} />}</div>}
    </section>
    <div className="ws-grid-2">
      <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Prepare for training</p><h2>Build a reliable foundation</h2></div><Icon name="layers" /></div><div className="ws-steps">
        <div className="ws-step-card"><span className="ws-step-num">01</span><div><h3 className="ws-step-title">Validate recordings</h3><p className="ws-step-desc">Check recording structure, metadata, and data provenance.</p><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={busy !== null} onClick={() => void prepare("validate")}>{busy === "validate" ? "Validating…" : validation ? "Run validation again" : "Run validation"}</button></div></div>
        <div className="ws-step-card"><span className="ws-step-num">02</span><div><h3 className="ws-step-title">Build the manifest</h3><p className="ws-step-desc">Index the available recordings and their existing metadata.</p><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={busy !== null || !validation?.ok} onClick={() => void prepare("manifest")}>{busy === "manifest" ? "Building…" : "Build manifest"}</button>{manifestEntries !== null && <p className="ws-note">{manifestEntries} entries indexed</p>}</div></div>
        <div className="ws-step-card"><span className="ws-step-num">03</span><div><h3 className="ws-step-title">Create dataset splits</h3><p className="ws-step-desc">Create training, validation, and test partitions by speaker.</p><button className="ws-btn ws-btn-ghost ws-btn-sm" disabled={busy !== null || manifestEntries === null || manifestEntries === 0} onClick={() => void prepare("splits")}>{busy === "splits" ? "Creating splits…" : "Create splits"}</button></div></div>
      </div></section>
      <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Dataset quality</p><h2>Validation report</h2></div><span className={`ws-badge ${validation?.ok ? "ws-badge-ready" : "ws-badge-muted"}`}>{validation ? validation.ok ? "Passed" : "Needs attention" : "Not checked"}</span></div>
        {validation ? <dl className="ws-detail-list">{validation.checks.map((check, index) => <div key={`${check.name}-${index}`}><dt><span className={`ws-status-dot ${check.pass ? "ready" : "warning"}`} />{check.name.replace(/_/g, " ")}</dt><dd>{check.detail || (check.pass ? "Passed" : "Failed")}</dd></div>)}</dl> : <div className="ws-empty"><Icon name="shield" size={32} /><h3>Know your data before training</h3><p>Run validation to see checks and actionable results from the dataset service.</p></div>}
        {message && <p className="ws-note" role="status">{message}</p>}{error && <p className="ws-error" role="alert">{error}</p>}{splits && <><dl className="ws-detail-list">{Object.entries(splits).map(([name, count]) => <div key={name}><dt>{name}</dt><dd>{count} recordings</dd></div>)}</dl><Link className="ws-btn ws-btn-primary" to="/demo/training">Open training lab <Icon name="arrow" /></Link></>}
      </section>
    </div>
  </div>
}
