import { useBackend } from '../hooks/useBackend'
import { Icon } from './Icon'

export function StatusPanel() {
  const { status, modelLoaded, capabilities } = useBackend()
  const online = status === 'online'

  return (
    <section className="ws-system-card ws-card-glow ws-hud-bracket" aria-label="System status">
      <div className="ws-system-title" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '7px' }}>
          <Icon name="cpu" size={14} />
          <span style={{ fontWeight: 600, letterSpacing: '0.04em' }}>Neural Engine</span>
        </div>
        <span style={{
          width: '7px',
          height: '7px',
          borderRadius: '50%',
          background: online ? '#10b981' : '#f43f5e',
          boxShadow: online ? '0 0 10px #10b981' : '0 0 8px #f43f5e',
          animation: online ? 'ws-pulse-emerald-anim 2s infinite' : 'none'
        }} />
      </div>

      <div className="ws-system-row" style={{ marginTop: '8px' }}>
        <span style={{ opacity: 0.7 }}>Inference API</span>
        <span style={{ color: online ? '#38bdf8' : '#94a3b8', fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>
          {online ? 'ONLINE · 127.0.0.1' : status === 'checking' ? 'PROBING…' : 'DISCONNECTED'}
        </span>
      </div>

      <div className="ws-system-row">
        <span style={{ opacity: 0.7 }}>Word Backbone</span>
        <span style={{
          color: capabilities?.avhubert_head?.ready ? '#10b981' : '#f59e0b',
          fontFamily: 'JetBrains Mono, monospace',
          fontSize: '9px',
          fontWeight: 600
        }}>
          {capabilities?.avhubert_head?.ready ? 'AV-HuBERT (L9)' : 'MIRACL-8 REF'}
        </span>
      </div>

      <div className="ws-system-row">
        <span style={{ opacity: 0.7 }}>Vision Pipeline</span>
        <span style={{ color: '#10b981', fontFamily: 'JetBrains Mono, monospace', fontSize: '9px' }}>
          MediaPipe Mesh (478 pts)
        </span>
      </div>

      <div className="ws-system-row">
        <span style={{ opacity: 0.7 }}>Letter Model</span>
        <span style={{ color: modelLoaded ? '#38bdf8' : '#94a3b8', fontFamily: 'JetBrains Mono, monospace', fontSize: '9px' }}>
          {modelLoaded ? 'BiLSTM 913k Param' : 'Unloaded'}
        </span>
      </div>

      <div style={{
        marginTop: '10px',
        paddingTop: '8px',
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '9px',
        fontFamily: 'JetBrains Mono, monospace',
        color: '#94a3b8'
      }}>
        <span>HARDWARE ACCEL</span>
        <span style={{ color: '#10b981' }}>CPU (MKL/OMP)</span>
      </div>
    </section>
  )
}
