/**
 * Workspace — Experiments
 *
 * Reads training checkpoints and groups them into runs for side-by-side
 * comparison. The brief calls for loss curves + a hyperparameter table;
 * the underlying source of truth is the runs/<run>/summary.json that
 * the training pipeline writes.
 */
import { useCallback, useEffect, useState } from "react"
import { trainingCheckpoints, type TrainingCheckpoint } from "../api"
import { API_BASE, fetchWithTimeout } from "../../lib/api"

interface RunSummary {
  name: string
  epochs: number
  best_val_accuracy: number
  best_epoch: number
  history: { epoch: number; train_loss: number; val_loss: number; val_accuracy: number }[]
  config_hash: string
}

function hyperparamTable(run: RunSummary): { key: string; value: string }[] {
  return [
    { key: "Run", value: run.name },
    { key: "Epochs", value: String(run.epochs) },
    { key: "Best val accuracy", value: run.best_val_accuracy.toFixed(4) },
    { key: "Best epoch", value: String(run.best_epoch) },
    { key: "Config hash", value: run.config_hash || "—" },
  ]
}

function MiniLossCurve({ history }: { history: RunSummary["history"] }) {
  if (history.length === 0) return <div className="ws-loss-plot" />
  const w = 240, h = 48
  const maxLoss = Math.max(...history.map((p) => Math.max(p.train_loss, p.val_loss)), 1)
  const minLoss = Math.min(...history.map((p) => Math.min(p.train_loss, p.val_loss)))
  const span = Math.max(1e-6, maxLoss - minLoss)
  const x = (i: number) => (history.length === 1 ? 0 : (i / (history.length - 1)) * w)
  const y = (v: number) => h - ((v - minLoss) / span) * h
  const trainPts = history.map((p, i) => `${x(i)},${y(p.train_loss)}`).join(" ")
  const valPts = history.map((p, i) => `${x(i)},${y(p.val_loss)}`).join(" ")
  return (
    <svg width={w} height={h} className="ws-loss-svg" aria-label="loss curve">
      <polyline points={trainPts} fill="none" stroke="var(--ws-accent)" strokeWidth="1.5" />
      <polyline points={valPts} fill="none" stroke="var(--ws-accent2)" strokeWidth="1.5" />
    </svg>
  )
}

export function ExperimentsPage() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    setLoading(true); setErr(null)
    const r = await trainingCheckpoints()
    if (!r) { setErr("Backend unreachable."); setLoading(false); return }
    const summaries: RunSummary[] = []
    for (const cp of r.checkpoints ?? []) {
      const s = await loadSummary(cp)
      if (s) summaries.push(s)
    }
    setRuns(summaries)
    setLoading(false)
  }, [])

  useEffect(() => { void load() }, [load])

  return (
    <div className="ws-page-inner">
      <header className="ws-page-head">
        <h1>Experiments</h1>
        <p className="ws-page-sub">Compare training runs by loss curve and hyperparameters.</p>
      </header>
      {err && <div className="ws-error" role="alert">{err}</div>}
      <div className="ws-action-row">
        <button className="ws-btn ws-btn-ghost" onClick={() => void load()}>{loading ? "Loading…" : "Refresh"}</button>
      </div>
      {loading ? <p className="ws-mono">Loading…</p>
        : runs.length === 0 ? (
          <div className="ws-empty">
            <p className="ws-empty-icon">🧪</p>
            <p>No experiments yet. Train a model to see it here.</p>
          </div>
        ) : (
          <div className="ws-exp-grid">
            {runs.map((r) => (
              <section key={r.name} className="ws-card ws-exp-card">
                <h2>{r.name}</h2>
                <MiniLossCurve history={r.history} />
                <table className="ws-table ws-exp-table">
                  <tbody>{hyperparamTable(r).map((row) => (
                    <tr key={row.key}><td>{row.key}</td><td className="ws-mono">{row.value}</td></tr>
                  ))}</tbody>
                </table>
              </section>
            ))}
          </div>
        )}
    </div>
  )

  async function loadSummary(cp: TrainingCheckpoint): Promise<RunSummary | null> {
    const summaryUrl = `${API_BASE}/api/training/checkpoints/${encodeURIComponent(cp.name)}/summary`
    try {
      const res = await fetchWithTimeout(summaryUrl, {}, 3000)
      if (!res.ok) return null
      const d = (await res.json()) as Partial<RunSummary>
      return {
        name: cp.name,
        epochs: d.epochs ?? cp.epoch ?? 0,
        best_val_accuracy: d.best_val_accuracy ?? cp.best_val ?? 0,
        best_epoch: d.best_epoch ?? 0,
        history: d.history ?? [],
        config_hash: d.config_hash ?? cp.config_hash ?? "",
      }
    } catch {
      return null
    }
  }
}