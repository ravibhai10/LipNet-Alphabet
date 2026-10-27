import { useCallback, useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { experimentSummaries, trainingCheckpoints, type TrainingCheckpoint, type ExperimentRun } from "../api"
import { Icon } from "../Icon"

export function ExperimentsPage() {
  const [runs, setRuns] = useState<TrainingCheckpoint[] | null>(null)
  const [loading, setLoading] = useState(true)
  const [query, setQuery] = useState("")
  const [selected, setSelected] = useState<string[]>([])
  const [summaries, setSummaries] = useState<ExperimentRun[] | null>(null)
  const [curvePath, setCurvePath] = useState("")
  const load = useCallback(async () => {
    setLoading(true)
    const [result, recorded] = await Promise.all([trainingCheckpoints(), experimentSummaries()])
    setRuns(result?.checkpoints ?? null)
    setSummaries(recorded?.runs ?? null)
    setLoading(false)
  }, [])
  useEffect(() => { void load() }, [load])
  const visible = (runs ?? []).filter(r => `${r.name} ${r.config_hash ?? ""}`.toLowerCase().includes(query.trim().toLowerCase()))
  const compared = (runs ?? []).filter(r => selected.includes(r.path))
  const curveRun = summaries?.find(run => run.path === curvePath) ?? summaries?.find(run => run.history.length > 0) ?? summaries?.[0]
  function toggle(path: string) { setSelected(current => current.includes(path) ? current.filter(p => p !== path) : [...current.slice(-1), path]) }

  return <div className="ws-page-inner">
    <header className="ws-page-header"><p className="ws-kicker">05 / Experiments</p><h1 className="ws-title">A clearer view of<br /><em>every iteration.</em></h1><p className="ws-lede">Explore recorded checkpoints and compare the metadata available from your training runs.</p></header>
    <div className="ws-toolbar"><label className="ws-field"><span>Find a training run</span><input className="ws-input" type="search" placeholder="Name or configuration hash…" value={query} onChange={e => setQuery(e.target.value)} /></label><button className="ws-btn ws-btn-ghost" disabled={loading} onClick={() => void load()}><Icon name="refresh" />{loading ? "Refreshing…" : "Refresh runs"}</button><Link className="ws-btn ws-btn-primary" to="/demo/training">Training lab <Icon name="arrow" /></Link></div>
    <div className="ws-grid-2">
      <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Run history</p><h2>Checkpoint-backed experiments</h2></div><Icon name="flask" /></div>
        {loading ? <p className="ws-note" role="status">Reading training runs…</p> : runs === null ? <div className="ws-empty"><Icon name="alert" size={30} /><h3>Training history unavailable</h3><p>Check your backend connection, then refresh the runs.</p></div> : visible.length === 0 ? <div className="ws-empty"><Icon name="flask" size={32} /><h3>{query ? "No matching runs" : "Room for your first experiment"}</h3><p>{query ? "Search another name or clear the filter." : "A completed training run creates a checkpoint you can inspect here."}</p></div> : <div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Compare</th><th>Run</th><th>Epoch</th></tr></thead><tbody>{visible.map(run => <tr key={run.path}><td><input type="checkbox" aria-label={`Compare ${run.name}`} checked={selected.includes(run.path)} onChange={() => toggle(run.path)} /></td><td><strong>{run.name}</strong><small className="ws-code">{run.created || "Creation date not provided"}</small></td><td>{run.epoch ?? "—"}</td></tr>)}</tbody></table></div>}
        <p className="ws-note">Select up to two checkpoints for a side-by-side comparison.</p>
      </section>
      <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Recorded training telemetry</p><h2>Learning curves</h2></div><span className="ws-badge ws-badge-muted">{curveRun?.history.length ? 'Saved history' : 'No history'}</span></div>{curveRun ? <><label className="ws-field"><span>Saved run</span><select className="ws-input" value={curveRun.path} onChange={event => setCurvePath(event.target.value)}>{summaries?.map(run => <option key={run.path} value={run.path}>{run.name}</option>)}</select></label><LossCurve run={curveRun}/><dl className="ws-detail-list"><dt>Recorded epochs</dt><dd>{curveRun.epochs}</dd><dt>Best saved validation accuracy</dt><dd>{curveRun.best_val_accuracy != null ? (curveRun.best_val_accuracy * 100).toFixed(1) + '%' : 'Not reported'}</dd><dt>Saved test accuracy</dt><dd>{curveRun.test_accuracy != null ? (curveRun.test_accuracy * 100).toFixed(1) + '%' : 'Not reported'}</dd></dl><p className="ws-note">Metrics come from the saved training report. They do not measure live webcam accuracy.</p></> : <div className="ws-empty"><Icon name="chart" size={34}/><h3>{summaries === null ? 'Training history unavailable' : 'No saved epoch history'}</h3><p>Recorded loss values will appear when a training report is available.</p></div>}</section>
    </div>
    {compared.length > 0 && <section className="ws-card"><div className="ws-card-head"><div><p className="ws-kicker">Compare metadata</p><h2>{compared.length === 1 ? "Choose another run to compare" : "Two runs. One view."}</h2></div><button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={() => setSelected([])}>Clear selection</button></div><div className="ws-table-wrap"><table className="ws-table"><thead><tr><th>Available metadata</th>{compared.map(r => <th key={r.path}>{r.name}</th>)}</tr></thead><tbody><tr><td>Epoch</td>{compared.map(r => <td key={r.path}>{r.epoch ?? "Not provided"}</td>)}</tr><tr><td>Best validation metric</td>{compared.map(r => <td key={r.path}>{r.best_val != null ? r.best_val.toFixed(4) : "Not provided"}</td>)}</tr><tr><td>Configuration hash</td>{compared.map(r => <td key={r.path} className="ws-code">{r.config_hash || "Not provided"}</td>)}</tr><tr><td>Representation</td>{compared.map(r => <td key={r.path}>{r.repr_version || "Not provided"}</td>)}</tr><tr><td>Vocabulary</td>{compared.map(r => <td key={r.path}>{r.vocab || "Not provided"}</td>)}</tr></tbody></table></div></section>}
  </div>
}

function LossCurve({ run }: { run: ExperimentRun }) {
  const points = run.history.filter(row => Number.isFinite(row.train_loss) || Number.isFinite(row.val_loss))
  if (!points.length) return <div className="ws-empty" style={{ marginTop: 16 }}><Icon name="chart" size={24}/><p>This report has no per-epoch loss values.</p></div>
  const values = points.flatMap(row => [row.train_loss, row.val_loss]).filter((value): value is number => typeof value === 'number' && Number.isFinite(value))
  const low = Math.min(...values), high = Math.max(...values), span = high - low || 1
  const x = (epoch: number) => 42 + (epoch - points[0].epoch) / Math.max(1, points[points.length - 1].epoch - points[0].epoch) * 500
  const y = (loss: number) => 153 - (loss - low) / span * 126
  const line = (key: 'train_loss' | 'val_loss') => points.filter(row => typeof row[key] === 'number' && Number.isFinite(row[key])).map(row => x(row.epoch) + ',' + y(row[key]!)).join(' ')
  return <figure className="ws-learning-curve"><svg viewBox="0 0 570 185" role="img" aria-label={'Saved training and validation loss for ' + run.name}><title>Recorded loss by epoch</title>{[0,.5,1].map(fraction => <g key={fraction}><line x1="42" x2="542" y1={27 + fraction * 126} y2={27 + fraction * 126} stroke="var(--ws-border)"/><text x="0" y={31 + fraction * 126} fill="var(--ws-text-muted)" fontSize="10">{(high - fraction * (high - low)).toFixed(2)}</text></g>)}<polyline points={line('train_loss')} fill="none" stroke="var(--ws-accent)" strokeWidth="2"/><polyline points={line('val_loss')} fill="none" stroke="var(--ws-text)" strokeWidth="1.5" strokeDasharray="5 4"/><text x="42" y="178" fill="var(--ws-text-muted)" fontSize="10">Epoch {points[0].epoch}</text><text x="500" y="178" fill="var(--ws-text-muted)" fontSize="10">{points[points.length - 1].epoch}</text></svg><figcaption><span>— Training loss</span><span>┄ Validation loss</span></figcaption></figure>
}
