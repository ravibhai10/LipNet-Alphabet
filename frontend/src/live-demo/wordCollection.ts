/** Labelled recordings are data, never model predictions. Kept locally in IndexedDB. */
export const EIGHT_WORDS = ['choose', 'connection', 'hello', 'navigation', 'next', 'previous', 'start', 'stop'] as const
export const TWENTY_WORDS = [
  'begin', 'bin', 'box', 'cancel', 'choose', 'close', 'connection', 'hello', 'help',
  'navigate', 'navigation', 'next', 'no', 'open', 'place', 'previous', 'set', 'start', 'stop', 'yes'
] as const
export const EXTRA_WORDS = ['help', 'yes', 'no', 'water', 'doctor', 'please', 'thanks'] as const
export const ALL_COLLECTION_WORDS = Array.from(new Set([...TWENTY_WORDS, ...EXTRA_WORDS])) as readonly string[]
const VOCABULARY_KEY = 'lipnet-eight-word-vocabulary-v1'
export type EightWordVocabulary = string[]
export function getEightWordVocabulary(): EightWordVocabulary {
  try {
    const raw = localStorage.getItem(VOCABULARY_KEY)
    const parsed = raw ? JSON.parse(raw) : null
    if (Array.isArray(parsed) && parsed.length === 8 && parsed.every(word => typeof word === 'string' && /^[a-z][a-z0-9_-]{0,31}$/.test(word))) {
      return [...new Set(parsed)] .length === 8 ? parsed : [...EIGHT_WORDS]
    }
  } catch { /* Use the verified default vocabulary when browser storage is unavailable. */ }
  return [...EIGHT_WORDS]
}
export function setEightWordVocabulary(words: string[]): EightWordVocabulary {
  const normalized = words.map(word => word.trim().toLowerCase())
  if (normalized.length !== 8 || new Set(normalized).size !== 8 || normalized.some(word => !/^[a-z][a-z0-9_-]{0,31}$/.test(word))) {
    throw new Error('Enter exactly 8 unique words using lowercase letters, numbers, underscores or hyphens.')
  }
  localStorage.setItem(VOCABULARY_KEY, JSON.stringify(normalized))
  return normalized
}
export function resetEightWordVocabulary() {
  localStorage.removeItem(VOCABULARY_KEY)
  return [...EIGHT_WORDS]
}
export type CollectionSplit = 'train' | 'valid' | 'test'
export interface WordRecording {
  id: string; label: string; speaker_id: string; session_id: string; split: CollectionSplit
  frames: string[]; timestamps_ms: number[]; recorded_at: string; duration_ms: number
  preprocessing: 'browser-eyeaxis96-jpeg-v1'; landmarker_sha256: string
  source: 'user_camera'; label_confirmed: true; consent: true; frames_sha256: string
}
export const COLLECTION_QUOTAS = { train: 15, valid: 5, test: 13 } as const
export function validateRecording(value: WordRecording, allowedWords: readonly string[] = EIGHT_WORDS) {
  if (allowedWords.length > 0 && !allowedWords.includes(value.label)) {
    throw new Error('Choose one supported word and dataset partition.')
  }
  if (!['train','valid','test'].includes(value.split)) throw new Error('Choose one supported dataset partition.')
  if (![value.id,value.speaker_id,value.session_id].every(v => /^[a-zA-Z0-9_-]{1,80}$/.test(v))) throw new Error('Use letters, numbers, underscores or hyphens for participant and session IDs.')
  if (!value.consent || !value.label_confirmed || value.source !== 'user_camera' || value.preprocessing !== 'browser-eyeaxis96-jpeg-v1') throw new Error('Consent and the spoken-word confirmation are required.')
  if (!/^[a-f0-9]{64}$/.test(value.landmarker_sha256) || !/^[a-f0-9]{64}$/.test(value.frames_sha256)) throw new Error('Recording fingerprint is missing.')
  if (value.frames.length < 8 || value.frames.length > 76 || value.frames.length !== value.timestamps_ms.length || !value.frames.every(f => f.startsWith('data:image/jpeg;base64,') && f.length < 200000)) throw new Error('Record 8–76 valid mouth frames.')
  if (value.timestamps_ms.some((t,i) => !Number.isFinite(t) || i > 0 && t <= value.timestamps_ms[i-1]) || !Number.isFinite(Date.parse(value.recorded_at))) throw new Error('Invalid recording timestamps.')
  if (!(value.duration_ms >= 0 && value.duration_ms <= 5000)) throw new Error('Keep one word within five seconds.')
}
export function validateCollection(rows: WordRecording[], vocabulary?: readonly string[]) {
  const ids = new Set<string>(), contents = new Set<string>(), sessions = new Map<string,CollectionSplit>()
  for (const r of rows) {
    validateRecording(r, vocabulary)
    const session = `${r.speaker_id}/${r.session_id}`
    if (ids.has(r.id) || contents.has(r.frames_sha256)) throw new Error('A duplicate recording is already saved.')
    if (sessions.has(session) && sessions.get(session) !== r.split) throw new Error('A recording session cannot cross dataset partitions. Start a new session ID.')
    ids.add(r.id); contents.add(r.frames_sha256); sessions.set(session,r.split)
  }
}
export function collectionCounts(rows: Pick<WordRecording,'label'|'split'>[], split: CollectionSplit, vocabulary: readonly string[] = EIGHT_WORDS) {
  return Object.fromEntries(vocabulary.map(word => [word,rows.filter(r=>r.label===word && r.split===split).length])) as Record<string,number>
}
export async function digest(bytes: BufferSource): Promise<string> {
  return [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(n=>n.toString(16).padStart(2,'0')).join('')
}
async function database(): Promise<IDBDatabase> {
  return new Promise((resolve,reject)=>{
    const request=indexedDB.open('lipnet-labelled-word-recordings-v1',1)
    request.onupgradeneeded=()=>request.result.createObjectStore('clips',{keyPath:'id'})
    request.onsuccess=()=>resolve(request.result)
    request.onerror=()=>reject(new Error('Local recording storage is unavailable. Allow browser storage and retry.'))
    request.onblocked=()=>reject(new Error('Close other collection tabs and retry.'))
  })
}
export async function readRecordings(): Promise<WordRecording[]> {
  const db=await database()
  try { return await new Promise((resolve,reject)=>{const tx=db.transaction('clips','readonly'),q=tx.objectStore('clips').getAll();q.onsuccess=()=>resolve(q.result);q.onerror=()=>reject(q.error)}) }
  finally { db.close() }
}
export async function saveRecording(clip: WordRecording, vocabulary?: readonly string[]) {
  validateCollection([...(await readRecordings()),clip], vocabulary)
  const db=await database()
  try { await new Promise<void>((resolve,reject)=>{const tx=db.transaction('clips','readwrite');tx.objectStore('clips').add(clip);tx.oncomplete=()=>resolve();tx.onerror=()=>reject(new Error('Could not save this recording. Storage may be full; export your data first.'));tx.onabort=()=>reject(new Error('Recording save was cancelled. Retry before leaving the page.'))}) }
  finally { db.close() }
}
export async function deleteRecording(id: string) {
  const db=await database()
  try { await new Promise<void>((resolve,reject)=>{const tx=db.transaction('clips','readwrite');tx.objectStore('clips').delete(id);tx.oncomplete=()=>resolve();tx.onerror=()=>reject(tx.error)}) }
  finally { db.close() }
}
