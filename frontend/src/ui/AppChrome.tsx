import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import './app-chrome.css'

type ToastTone = 'success' | 'error' | 'info'
type Toast = { id: number; tone: ToastTone; message: string }

interface UiContextValue {
  toast: (message: string, tone?: ToastTone) => void
  audioEnabled: boolean
  setAudioEnabled: (enabled: boolean) => void
}

const UiContext = createContext<UiContextValue>({
  toast: () => {},
  audioEnabled: false,
  setAudioEnabled: () => {},
})

const destinations = [
  ['/demo/overview', 'Overview'],
  ['/demo/live', 'Live lip reading'],
  ['/demo/collect', 'Record word data'],
  ['/demo/dataset', 'Dataset explorer'],
  ['/demo/training', 'Training'],
  ['/demo/testing', 'Evaluation and testing'],
  ['/demo/models', 'Model registry'],
  ['/demo/docs', 'Documentation'],
  ['/science', 'Science Lab'],
  ['/research', 'Research exhibits'],
] as const

function readStorage(key: string, fallback: string) {
  try { return localStorage.getItem(key) || fallback } catch { return fallback }
}

export function useUi() { return useContext(UiContext) }

export function AppChrome({ children }: { children: ReactNode }) {
  const navigate = (path: string) => window.location.assign(path)
  const [theme] = useState(() => readStorage('lipnet.theme', 'dark'))
  const [scale] = useState(() => readStorage('lipnet.fontScale', 'medium'))
  const [contrast] = useState(() => readStorage('lipnet.contrast', 'normal') === 'high')
  const [audioEnabled, setAudioEnabled] = useState(() => readStorage('lipnet.audio', 'off') === 'on')
  const [online, setOnline] = useState(navigator.onLine)
  const [toasts, setToasts] = useState<Toast[]>([])
  const [paletteOpen, setPaletteOpen] = useState(false)
  const [feedbackOpen, setFeedbackOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [feedback, setFeedback] = useState('')

  const persist = useCallback((key: string, value: string) => {
    try { localStorage.setItem(key, value) } catch { /* private browsing */ }
  }, [])
  const toast = useCallback((message: string, tone: ToastTone = 'info') => {
    const id = Date.now() + Math.random()
    setToasts(current => [...current, { id, message, tone }])
    window.setTimeout(() => setToasts(current => current.filter(item => item.id !== id)), 4500)
  }, [])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    document.documentElement.dataset.fontScale = scale
    document.documentElement.dataset.contrast = contrast ? 'high' : 'normal'
    persist('lipnet.theme', theme); persist('lipnet.fontScale', scale); persist('lipnet.contrast', contrast ? 'high' : 'normal')
  }, [theme, scale, contrast, persist])
  useEffect(() => {
    const onlineChange = () => setOnline(true)
    const offlineChange = () => setOnline(false)
    window.addEventListener('online', onlineChange); window.addEventListener('offline', offlineChange)
    return () => { window.removeEventListener('online', onlineChange); window.removeEventListener('offline', offlineChange) }
  }, [])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault(); setPaletteOpen(true)
      }
      if (event.key === 'Escape') { setPaletteOpen(false); setFeedbackOpen(false) }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  const matches = useMemo(() => destinations.filter(([, label]) => label.toLowerCase().includes(query.toLowerCase())), [query])
  const value = useMemo(() => ({ toast, audioEnabled, setAudioEnabled: (enabled: boolean) => { setAudioEnabled(enabled); persist('lipnet.audio', enabled ? 'on' : 'off') } }), [toast, audioEnabled, persist])

  return (
    <UiContext.Provider value={value}>
      <div className="app-chrome">
        {!online && <div className="app-offline-banner" role="status">Local Offline Mode Active · Network connection unavailable</div>}
        {children}
        <div className="app-global-tools" aria-label="Project identity">
          <span style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "6px",
            border: "1px solid rgba(59, 130, 246, 0.4)",
            background: "rgba(15, 23, 42, 0.92)",
            color: "#60a5fa",
            borderRadius: "8px",
            padding: "7px 14px",
            fontSize: "12px",
            fontWeight: 600,
            letterSpacing: "0.04em",
            boxShadow: "0 4px 12px rgba(0, 0, 0, 0.4)",
            userSelect: "none"
          }}>
            <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: "#60a5fa" }} />
            Group 12
          </span>
        </div>
        <div className="app-toasts" aria-live="polite" aria-atomic="true">
          {toasts.map(item => <div key={item.id} className={`app-toast app-toast-${item.tone}`} role={item.tone === 'error' ? 'alert' : 'status'}>{item.message}<button type="button" onClick={() => setToasts(current => current.filter(toastItem => toastItem.id !== item.id))} aria-label="Dismiss notification">×</button></div>)}
        </div>
        {paletteOpen && <div className="app-modal-backdrop" onMouseDown={event => event.currentTarget === event.target && setPaletteOpen(false)}>
          <section className="app-palette" role="dialog" aria-modal="true" aria-labelledby="palette-title">
            <h2 id="palette-title">Jump to a workspace</h2>
            <input autoFocus value={query} onChange={event => setQuery(event.target.value)} placeholder="Search pages, checkpoints, letters…" aria-label="Search workspace" />
            <div className="app-palette-list">{matches.map(([path, label]) => <button type="button" key={path} onClick={() => { navigate(path); setPaletteOpen(false); setQuery('') }}><span>{label}</span><kbd>↵</kbd></button>)}</div>
            <p>Use <kbd>Ctrl</kbd>+<kbd>K</kbd> anywhere · <kbd>Esc</kbd> to close</p>
          </section>
        </div>}
        {feedbackOpen && <div className="app-modal-backdrop" onMouseDown={event => event.currentTarget === event.target && setFeedbackOpen(false)}>
          <section className="app-feedback" role="dialog" aria-modal="true" aria-labelledby="feedback-title">
            <h2 id="feedback-title">Feedback & bug report</h2>
            <p>Record a note for this demonstration. It stays in this browser.</p>
            <textarea autoFocus value={feedback} onChange={event => setFeedback(event.target.value)} placeholder="What did you observe?" aria-label="Feedback note" />
            <div><button type="button" className="app-primary" onClick={() => { if (feedback.trim()) { persist('lipnet.feedback.last', feedback.trim()); toast('Feedback note saved locally.', 'success'); setFeedback(''); setFeedbackOpen(false) } }}>Save note</button><button type="button" onClick={() => setFeedbackOpen(false)}>Cancel</button></div>
          </section>
        </div>}
      </div>
    </UiContext.Provider>
  )
}
