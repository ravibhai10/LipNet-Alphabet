import { useBackend } from "../hooks/useBackend"
import { useEffect, useState } from "react"
import { inferenceCheckpoints, liveState, type InferenceCheckpoint, type LiveState } from "./api"

export function StatusPanel() {
  const { status, modelLoaded, datasetStats } = useBackend()
  const [inference, setInference] = useState<InferenceCheckpoint[] | null>(null)
  const [live, setLive] = useState<LiveState | null>(null)
  const [hasCamera, setHasCamera] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    void (async () => {
      const inf = await inferenceCheckpoints()
      if (!cancelled) setInference(inf?.checkpoints ?? [])
    })()
    return () => { cancelled = true }
  }, [])

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const tick = async () => {
      const s = await liveState()
      if (!cancelled) setLive(s)
    }
    void tick()
    timer = window.setInterval(tick, 3000)
    return () => { cancelled = true; if (timer) window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    // Best-effort camera probe (do not fabricate if not present)
    if (navigator?.mediaDevices?.getUserMedia) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((s) => { setHasCamera(true); s.getTracks().forEach((t) => t.stop()) })
        .catch(() => setHasCamera(false))
    } else {
      setHasCamera(false)
    }
  }, [])

  // ── Status logic ──────────────────────────────────────────────────────────
  const mediapipeStatus = live !== null || inference !== null ? "ready" : "neutral"
  const modelStatus =
    status === "online" && (modelLoaded === true || (inference && inference.length > 0))
      ? "ready"
      : status === "online"
        ? "warning"
        : status === "checking"
          ? "processing"
          : "neutral"
  const modelText =
    status === "online"
      ? modelLoaded === true
        ? "READY"
        : inference && inference.length > 0
          ? "CHECKPOINTS"
          : "NOT TRAINED"
      : "OFFLINE"
  const cameraStatus: "ready" | "warning" | "neutral" =
    hasCamera === true ? "ready" : hasCamera === false ? "warning" : "neutral"
  const datasetStatus =
    datasetStats && Object.keys(datasetStats).length > 0 ? "ready" : "neutral"
  const datasetText = datasetStats?.recordings
    ? `${datasetStats.recordings} rec`
    : datasetStats?.letters
      ? `${datasetStats.letters} L`
      : "—"

  return (
    <div className="ws-status" role="status" aria-label="System status">
      <div className="ws-status-label">System</div>
      <div className="ws-status-row">
        <span className="ws-status-key">Backend</span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`ws-status-dot ${status === "online" ? "ready" : status === "checking" ? "processing" : "neutral"}`} />
          {status.toUpperCase()}
        </span>
      </div>
      <div className="ws-status-row">
        <span className="ws-status-key">MediaPipe</span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`ws-status-dot ${mediapipeStatus}`} />
          {mediapipeStatus === "ready" ? "READY" : "—"}
        </span>
      </div>
      <div className="ws-status-row">
        <span className="ws-status-key">Model</span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`ws-status-dot ${modelStatus}`} />
          {modelText}
        </span>
      </div>
      <div className="ws-status-row">
        <span className="ws-status-key">Camera</span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`ws-status-dot ${cameraStatus}`} />
          {hasCamera === true ? "AVAILABLE" : hasCamera === false ? "BLOCKED" : "PROBING"}
        </span>
      </div>
      <div className="ws-status-row">
        <span className="ws-status-key">Dataset</span>
        <span style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <span className={`ws-status-dot ${datasetStatus}`} />
          {datasetText}
        </span>
      </div>
    </div>
  )
}
