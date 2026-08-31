import { PIPELINE_STEPS } from "../data/architecture"

interface Props {
  activeStage?: number  // 0-indexed into PIPELINE_STEPS
  activeLabel?: string // alternative: match by id
}

const PIPE_DIMS: Record<string, string> = {
  video: "30 fps",
  landmarks: "478 pts",
  repr: "T×96 / T×64",
  fusion: "T×192",
  conformer: "T×192",
  ctc: "T×27",
  letters: "26",
}

type PipeStep = (typeof PIPELINE_STEPS)[number]

export function PipelineFlow({ activeStage, activeLabel }: Props) {
  const activeIdx = activeLabel
    ? PIPELINE_STEPS.findIndex((s: PipeStep) => s.id === activeLabel)
    : activeStage ?? -1

  return (
    <div className="ws-pipeline-flow" role="list" aria-label="Processing pipeline">
      {PIPELINE_STEPS.map((step: PipeStep, i: number) => {
        const isActive = i <= activeIdx && activeIdx >= 0
        return (
          <div key={step.id} style={{ display: "flex", alignItems: "center" }} role="listitem">
            <div
              className={`ws-pipe-node ${isActive ? "active" : ""}`}
              title={step.note}
              aria-label={`${step.label}: ${step.note}${isActive ? " (active)" : ""}`}
            >
              <span className="ws-pipe-node-label">{step.label}</span>
              {PIPE_DIMS[step.id] && (
                <span className="ws-pipe-node-dim">{PIPE_DIMS[step.id]}</span>
              )}
            </div>
            {i < PIPELINE_STEPS.length - 1 && (
              <span className="ws-pipe-arrow" aria-hidden="true">→</span>
            )}
          </div>
        )
      })}
    </div>
  )
}
