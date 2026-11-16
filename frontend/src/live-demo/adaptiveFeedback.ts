import { API_BASE, fetchWithTimeout, responseError } from '../lib/api'

export type FeedbackTask = 'alphabet_recognition' | 'isolated_word'
export type FeedbackState = 'IDLE'|'CAPTURING'|'PROCESSING'|'PREDICTED'|'AWAITING_FEEDBACK'|'FEEDBACK_SAVED'|'UNCERTAIN'|'REJECTED'|'ERROR'
export interface AdaptivePrediction {
  prediction: string|null; candidate: string; confidence: number; model_id: string; model_version: string
  top_k: {label:string;score:number}[]; latency_ms: number|null; status: string
}
export interface FeedbackRecord {
  id:string;task:FeedbackTask;speaker:string;session:string;status:string;label:string|null
  verdict:string|null;created:string;frame_count:number;captured_fps:number;duration_ms:number
  prediction:AdaptivePrediction|null;warnings:string[];media_available:boolean;capture_kind:string
  metadata?:{timestamps_ms:number[]};prediction_error?:string
  original_video_available?:boolean
}
export interface AdaptiveConfig {
  tasks:Record<FeedbackTask,string[]>;correction_labels:Record<FeedbackTask,string[]>
  role:string;recording_consent:string
}
export async function adaptiveApi<T>(key:string,path:string,body?:unknown,signal?:AbortSignal):Promise<T> {
  const res=await fetchWithTimeout(`${API_BASE}/api/adaptive${path}`,{
    method:body===undefined?'GET':'POST',headers:{Authorization:`Bearer ${key}`,...(body===undefined?{}:{'Content-Type':'application/json'})},
    ...(body===undefined?{}:{body:JSON.stringify(body)}),signal},130000)
  if(!res.ok)throw await responseError(res)
  return res.json() as Promise<T>
}
export function feedbackDecision(prediction:AdaptivePrediction,verdict:'correct'|'incorrect',label:string,labels:string[]) {
  if(!label || ![...labels,'UNKNOWN'].includes(label))throw new Error('Choose the actual spoken label or UNKNOWN / OTHER.')
  if(verdict==='correct' && (!prediction.prediction || label!==prediction.prediction))throw new Error('An uncertain result needs an explicit corrected label.')
  if(verdict==='incorrect' && label===prediction.prediction)throw new Error('Choose a label different from the prediction.')
  return {verdict,label}
}
export function validSourceWindow(times:number[],bytes:number) {
  if(times.length<8||times.length>76)throw new Error('Record 8–76 complete source frames.')
  if(times.some((t,i)=>!Number.isFinite(t)||t<0||i>0&&(t<=times[i-1]||t-times[i-1]>250)))throw new Error('The recording contains a frame gap. Retry.')
  const duration=times[times.length-1]-times[0]
  if(duration>5000||(times.length-1)*1000/duration<8)throw new Error('Record at least eight source samples per second within five seconds.')
  if(bytes>90*1024*1024)throw new Error('Recording is too large. Record a shorter utterance.')
}
