import { Icon } from "./Icon"

export const REFERENCE_STAGES = [
  { id: "video", label: "Video input", dimension: "Browser camera", status: "Implemented", note: "Camera access is requested when you start a session. Frames are captured locally in the browser." },
  { id: "landmarks", label: "Face landmarks", dimension: "MediaPipe", status: "Implemented", note: "The browser detects facial landmarks and derives the mouth region when a face is visible." },
  { id: "temporal", label: "Frame sequence", dimension: "Ordered crops", status: "Implemented", note: "A bounded sequence of mouth crops is sent to the alphabet prediction endpoint." },
  { id: "repr", label: "Representation", dimension: "Research design", status: "Unverified", note: "Appearance, geometry, and motion streams are part of the research architecture. Their use depends on the loaded checkpoint." },
  { id: "fusion", label: "Gated fusion", dimension: "Research design", status: "Unverified", note: "Multistream gating belongs to the research model. The current runtime does not guarantee that fusion weights are available." },
  { id: "conformer", label: "Temporal encoder", dimension: "Model-dependent", status: "Unverified", note: "The research design includes a Conformer encoder. The live alphabet service may use a different checkpoint architecture." },
  { id: "ctc", label: "Decoder", dimension: "Model-dependent", status: "Unverified", note: "CTC is part of the research sequence decoder. Alphabet classifiers can instead return a single class probability distribution." },
  { id: "letters", label: "Alphabet output", dimension: "A–Z", status: "Implemented", note: "The live interface displays the actual returned letter, confidence, and available candidate probabilities." },
] as const

export function PipelineFlow({ activeStage = -1, activeLabel, onSelect }: { activeStage?: number; activeLabel?: string; onSelect?: (index: number) => void }) {
  const selected = activeLabel ? REFERENCE_STAGES.findIndex(s => s.id === activeLabel) : activeStage
  return <div className="ws-pipeline-flow" role="list" aria-label="Visual speech architecture reference">
    {REFERENCE_STAGES.map((step, index) => <div key={step.id} className="ws-pipe-item" role="listitem">
      {onSelect ? <button className={`ws-pipe-node ${index === selected ? "active" : ""}`} onClick={() => onSelect(index)} aria-pressed={index === selected}><span className="ws-pipe-node-label">{step.label}</span><span className="ws-pipe-node-dim">{step.dimension}</span></button> : <div className={`ws-pipe-node ${index === selected ? "active" : ""}`} title={step.note}><span className="ws-pipe-node-label">{step.label}</span><span className="ws-pipe-node-dim">{step.dimension}</span></div>}
      {index < REFERENCE_STAGES.length - 1 && <span className="ws-pipe-arrow" aria-hidden="true"><Icon name="chevron" size={14} /></span>}
    </div>)}
  </div>
}
