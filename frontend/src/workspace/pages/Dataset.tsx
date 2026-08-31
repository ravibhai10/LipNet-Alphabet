import { useState, useCallback } from "react"
import { validateDataset, buildManifest, createSplits } from "../api"
import { PipelineFlow } from "../PipelineFlow"
type Stage = "idle" | "validating" | "manifest" | "splits" | "done" | "error"
interface Validation { ok: boolean; checks: { name: string; pass: boolean; detail?: string }[] }
export function DatasetPage() {
  const [stage, setStage] = useState<Stage>("idle")
  const [validation, setValidation] = useState<Validation | null>(null)
  const [msg, setMsg] = useState<string>("")
  const [error, setError] = useState<string>("")
  const [progress, setProgress] = useState(0)
  const runValidation = useCallback(async () => {
    setStage("validating"); setProgress(0)
    const t = setInterval(() => setProgress(p => Math.min(p + 8, 90)), 400)
    const result = await validateDataset()
    clearInterval(t); setProgress(100)
    if (result) { setValidation(result); setStage(result.ok ? "manifest" : "error") }
    else { setStage("error"); setError("Could not reach backend") }
  }, [])
  const runManifest = useCallback(async () => {
    setStage("manifest"); setMsg("Building manifest…")
    const result = await buildManifest()
    if (result) { setMsg(result.ok ? "Manifest built successfully." : "Error: " + (result.note ?? "unknown")); setStage(result.ok ? "splits" : "error") }
    else { setStage("error"); setError("Could not reach backend") }
  }, [])
  const runSplits = useCallback(async () => {
    setStage("splits"); setMsg("Creating train/val/test splits…")
    const result = await createSplits()
    if (result) { setMsg(result.ok ? "Splits created." : "Error: " + (result.note ?? "unknown")); setStage(result.ok ? "done" : "error") }
    else { setStage("error"); setError("Could not reach backend") }
  }, [])
  const pipeStage = stage === "idle" ? -1 : stage === "validating" ? 0 : stage === "manifest" ? 2 : stage === "splits" ? 3 : stage === "done" ? 6 : -1
  return (
    <div>
      <div className="ws-page-header">
        <p className="ws-kicker">02 — Data</p>
        <h1 className="ws-title">Dataset Lab</h1>
        <p className="ws-lede">Validate your GRID corpus download, build the landmark manifest, and create train/val/test splits.</p>
      </div>
      <div className="ws-card" style={{ marginBottom: "20px" }}>
        <div className="ws-card-title">Pipeline Stage</div>
        <PipelineFlow activeStage={pipeStage} />
      </div>
      <div className="ws-grid-2">
        <div className="ws-card">
          <div className="ws-card-title">Preparation Steps</div>
          <div style={{ display: "flex", flexDirection: "column", gap: "12px", marginTop: "16px" }}>
            <div className="ws-step-card"><div className="ws-step-num">1</div><div>
              <div className="ws-step-title">Validate Download</div>
              <div className="ws-step-desc">Check GRID corpus directory structure and file integrity.</div>
              {stage === "idle" && <button className="ws-btn ws-btn-primary ws-btn-sm" style={{ marginTop: "8px" }} onClick={runValidation}>Run Validation</button>}
              {stage === "validating" && <div className="ws-progress-bar-wrap"><div className="ws-progress-bar-fill" style={{ width: progress + "%" }}/></div>}
            </div></div>
            <div className={"ws-step-card" + (validation?.ok ? "" : " dim")}><div className="ws-step-num">2</div><div>
              <div className="ws-step-title">Build Manifest</div>
              <div className="ws-step-desc">Index all video files and extract face landmarks with MediaPipe.</div>
              {validation?.ok && stage === "manifest" && <button className="ws-btn ws-btn-primary ws-btn-sm" style={{ marginTop: "8px" }} onClick={runManifest}>Build Manifest</button>}
              {(stage === "manifest" || stage === "splits" || stage === "done") && msg && <div className="ws-status-msg">{msg}</div>}
            </div></div>
            <div className="ws-step-card"><div className="ws-step-num">3</div><div>
              <div className="ws-step-title">Create Splits</div>
              <div className="ws-step-desc">Split data 80/10/10 by speaker for train/val/test.</div>
              {stage === "splits" && <button className="ws-btn ws-btn-primary ws-btn-sm" style={{ marginTop: "8px" }} onClick={runSplits}>Create Splits</button>}
            </div></div>
          </div>
        </div>
        <div className="ws-card">
          <div className="ws-card-title">Validation Results</div>
          {validation ? (
            <div style={{ marginTop: "16px" }}>{validation.checks.map((c, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: "10px", padding: "8px 0", borderBottom: "1px solid var(--ws-border)" }}>
                <span className={"ws-status-dot " + (c.pass ? "ready" : "warning")} />
                <span style={{ fontWeight: 500 }}>{c.name}</span>
                {c.detail && <span style={{ color: "var(--ws-text-dim)", fontSize: "12px" }}>{c.detail}</span>}
              </div>
            ))}</div>
          ) : <p style={{ color: "var(--ws-text-dim)", marginTop: "16px" }}>Run validation to see results.</p>}
          {stage === "done" && <div className="ws-badge ws-badge-ready" style={{ marginTop: "16px" }}>Dataset ready for training</div>}
          {stage === "error" && <div className="ws-badge ws-badge-warning" style={{ marginTop: "16px" }}>{error || "An error occurred"}</div>}
        </div>
      </div>
    </div>
  )
}
