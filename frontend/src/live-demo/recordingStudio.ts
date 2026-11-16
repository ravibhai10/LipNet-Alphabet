/** Studio records are labelled data. Model predictions cannot supply labels. */
export type StudioTask = 'alphabet_recognition' | 'isolated_word' | 'negative'
export type StudioSplit = 'train' | 'valid' | 'test'
export interface StudioConfig {
  configuration_sha256: string; landmarker_sha256: string; preprocessing: string
  tasks: Record<StudioTask, { labels: string[]; instruction: string }>
  collection_recommendation: string; final_test_rule: string
}
export interface StudioRecording {
  id: string; task: StudioTask; label: string; speaker_id: string; session_id: string; split: StudioSplit
  consent: boolean; label_confirmed: boolean; recorded_at: string; source: 'user_camera'
  capture_kind: 'physical_webcam' | 'webcam_replay'; preprocessing: string; landmarker_sha256: string
  frames: string[]; timestamps_ms: number[]; duration_ms: number; frames_sha256: string; frame_count: number
  spoken_text?: string
  camera: { width: number; height: number; captured_fps: number }
  conditions: { lighting: string; distance: string; speed: string; orientation: string }
  quality: { status: 'passed'; face_counts: number[]; mouth_valid: boolean[] }
  timing: { detection_ms: number[]; roi_ms: number[]; jpeg_ms: number[] }
}
export interface StudioClass {
  label: string; samples: number; speakers: number; splits: Record<StudioSplit,number>
  missing_splits: StudioSplit[]; recommendation: string; diversity_status: string; quality_warnings: number
}
export interface StudioInventory {
  samples: number; unique_speakers: number; tasks: Record<StudioTask, { labels: StudioClass[]; samples: number
    structural_training_eligible: boolean; reason: string; training_authorized: boolean }>
}
export function measuredFps(times: number[]) {
  if (times.length < 2 || times.some((t,i)=>!Number.isFinite(t) || t<0 || i>0 && t<=times[i-1])) return null
  return (times.length-1)*1000/(times[times.length-1]-times[0])
}
export function checkStudioRecording(row: StudioRecording, config: StudioConfig) {
  if (!row.consent || !row.label_confirmed) throw new Error('Consent and spoken-label confirmation are required.')
  if (!config.tasks[row.task]?.labels.includes(row.label)) throw new Error('Choose an explicit label for this task.')
  for (const id of [row.id,row.speaker_id,row.session_id]) if (!/^[a-zA-Z0-9_-]{1,80}$/.test(id)) throw new Error('Use valid recording, participant and session IDs.')
  const fps=measuredFps(row.timestamps_ms)
  if (row.frames.length<8 || row.frames.length>76 || row.frames.length!==row.timestamps_ms.length || !fps || fps<8 || row.frame_count!==row.frames.length || Math.abs(row.camera.captured_fps-fps)>.05) throw new Error('Capture 8–76 valid frames with measured FPS.')
  if (row.timestamps_ms.some((t,i)=>i>0 && t-row.timestamps_ms[i-1]>250)) throw new Error('Video capture stalled. Retry a complete utterance.')
  if (row.preprocessing!==config.preprocessing || row.landmarker_sha256!==config.landmarker_sha256 || row.quality.status!=='passed') throw new Error('Recording quality or preprocessing identity is missing.')
}
export function checkSpeakerAssignment(rows: Pick<StudioRecording,'speaker_id'|'split'>[],speaker: string,split: StudioSplit) {
  if (rows.some(r=>r.speaker_id===speaker && r.split!==split)) throw new Error('A participant cannot cross dataset partitions, even in different sessions.')
}
export const ACCEPTANCE_CASES = ['known_word','repeated_word','unknown_word','silence','permission_granted','permission_denied','camera_unavailable','no_face','multiple_faces','face_outside_frame','mouth_occlusion','low_lighting','bright_lighting','head_movement','near_distance','far_distance','short_utterance','long_utterance','camera_disconnection'] as const
export function downloadJson(value: unknown, name: string) {
  const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}))
  const link=document.createElement('a');link.href=url;link.download=name;link.click()
  setTimeout(()=>URL.revokeObjectURL(url),1000)
}
export async function verifyPlayback(frames: string[]) {
  for (const source of frames) {
    const img=new Image();img.src=source
    await img.decode()
    if (img.naturalWidth!==96 || img.naturalHeight!==96) throw new Error('Saved mouth frames cannot be decoded at 96px.')
  }
}
