export type SocketState = 'idle' | 'connecting' | 'connected' | 'degraded'
export function inferenceSocketUrl(apiBase: string, pageUrl: string, configured?: string) {
  const url = new URL(configured || apiBase || pageUrl, pageUrl)
  url.protocol = ['https:', 'wss:'].includes(url.protocol) ? 'wss:' : 'ws:'
  const prefix = configured ? url.pathname.replace(/\/$/, '') : ''
  url.pathname = prefix.endsWith('/inference') ? prefix : prefix.endsWith('/ws') ? prefix + '/inference' : prefix + '/ws/inference'
  url.search = ''; url.hash = ''
  return url.href
}

/** Optional legacy transport. Intentional close cancels all reconnect work. */
export class InferenceSocket {
  private socket: WebSocket | null = null
  private retry: ReturnType<typeof setTimeout> | undefined
  private ping: ReturnType<typeof setInterval> | undefined
  private enabled = false
  private attempts = 0
  private url: string
  private onState: (state: SocketState) => void
  private onMessage: (data: unknown) => void
  private create: (url: string) => WebSocket
  constructor(url: string, onState: (state: SocketState) => void, onMessage: (data: unknown) => void,
              create: (url: string) => WebSocket = url => new WebSocket(url)) {
    this.url = url; this.onState = onState; this.onMessage = onMessage; this.create = create
  }
  connect() {
    if (this.enabled) return
    this.enabled = true; this.attempts = 0; this.open()
  }
  private open() {
    if (!this.enabled || this.socket) return
    this.onState('connecting')
    const socket = this.create(this.url); this.socket = socket
    socket.onopen = () => {
      if (!this.enabled || this.socket !== socket) return
      this.onState('connected')
      this.ping = setInterval(() => { if (socket.readyState === 1) socket.send(JSON.stringify({ type: 'ping', ts: Date.now() })) }, 10000)
    }
    socket.onmessage = event => {
      if (!this.enabled || this.socket !== socket) return
      try { this.onMessage(JSON.parse(String(event.data))) } catch { this.onState('degraded') }
    }
    socket.onerror = () => { if (this.enabled && this.socket === socket) this.onState('degraded') }
    socket.onclose = () => {
      if (this.socket !== socket) return
      clearInterval(this.ping); this.ping = undefined; this.socket = null
      if (!this.enabled) return
      this.onState('degraded')
      if (this.attempts < 5) {
        const delay = Math.min(1000 * 2 ** this.attempts++, 15000)
        this.retry = setTimeout(() => { this.retry = undefined; this.open() }, delay)
      }
    }
  }
  sendFrame(frame: string) {
    if (!this.enabled || !this.socket || this.socket.readyState !== 1 || this.socket.bufferedAmount > 128000) return false
    this.socket.send(JSON.stringify({ type: 'frame', frame })); return true
  }
  disconnect() {
    this.enabled = false; clearTimeout(this.retry); clearInterval(this.ping)
    this.retry = this.ping = undefined
    const socket = this.socket; this.socket = null
    if (socket) { socket.onopen = socket.onclose = socket.onmessage = socket.onerror = null; socket.close() }
    this.onState('idle')
  }
}
