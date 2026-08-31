/**
 * Real architecture values — sourced from the actual project code:
 *   configs/alphabet_dataset.yaml, repr/appearance.py, repr/geometry.py,
 *   repr/motion.py, repr/fusion.py, temporal/encoder.py, alphabet/vocab.py.
 * Nothing in this file is invented. Do not add metrics here.
 */

export const APPEARANCE = {
  crop: '64 × 64 × 3',
  cropSize: [64, 64] as const,
  stemChannels: [32, 64, 128],
  outputDim: 96,
  label: 'Appearance',
  contribution: 0.55,
} as const

export const GEOMETRY = {
  landmarks: 20,
  scalars: 20,
  mlpHidden: 96,
  outputDim: 64,
  label: 'Geometry',
  contribution: 0.25,
  mouthLandmarkIndices: [
    78, 74, 81, 13, 312, 308, 402, 14, 61, 291,
    39, 37, 0, 267, 269, 391, 61, 0, 17, 291,
  ],
} as const

export const MOTION = {
  savgolWindow: 5,
  savgolPolyOrder: 2,
  flowPatch: [16, 16] as const,
  outputDim: 64,
  label: 'Motion',
  contribution: 0.20,
} as const

export const FUSION = {
  appearanceDim: 96,
  geometryDim: 64,
  motionDim: 64,
  contextDim: 64,
  outputDim: 192,
  label: 'Gated Fusion',
} as const

export const CONFORMER = {
  dModel: 192,
  blocks: 4,
  heads: 4,
  ffn: 768,
  convKernel: 7,
  dropout: 0.1,
  positionalEncoding: 'sinusoidal + learned',
} as const

export const DECODER = {
  classes: 27, // CTC blank + A–Z
  letters: 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split(''),
  blank: '∅',
} as const

export const SEQUENCE = {
  fps: 30,
  maxFrames: 76, // ≈ 2.5 s at 30 fps
  minFrames: 15,
} as const

export const SPLITS = {
  train: 70,
  validation: 15,
  test: 15,
  unit: '% by speaker',
} as const

export const UNCERTAINTY = {
  // From configs/alphabet_dataset.yaml decoder.uncertainty:
  // thresholds are fit on VALIDATION data only; temperature fitted by NLL.
  metrics: ['Prediction', 'Confidence', 'Margin', 'Entropy'] as const,
  states: ['CONFIDENT', 'AMBIGUOUS', 'ABSTAIN'] as const,
} as const

export const PIPELINE_STEPS = [
  { id: 'video', label: 'Real video', note: 'webcam / local mp4 · 30 fps' },
  { id: 'landmarks', label: 'Real MediaPipe landmarks', note: 'FaceMesh · 478 points · 20 canonical mouth points' },
  { id: 'temporal', label: 'Temporal data', note: 'ordered frames · 15–76 per recording' },
  { id: 'repr', label: 'Multistream representation', note: 'appearance · geometry · motion' },
  { id: 'fusion', label: 'Gated fusion', note: 'per-timestep α · 192-D' },
  { id: 'conformer', label: 'Conformer', note: '4 blocks · d_model 192 · heads 4 · FFN 768 · kernel 7' },
  { id: 'ctc', label: 'CTC', note: 'blank + 26 letters' },
  { id: 'letters', label: 'A–Z', note: 'spoken alphabet characters' },
] as const
