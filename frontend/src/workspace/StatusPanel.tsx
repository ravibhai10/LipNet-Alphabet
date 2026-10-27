import { useBackend } from '../hooks/useBackend'
import { Icon } from './Icon'
export function StatusPanel() {
  const { status, modelLoaded } = useBackend()
  return <section className="ws-system-card" aria-label="System status">
    <div className="ws-system-title"><Icon name="cpu" size={15}/><span>System connection</span></div>
    <div className="ws-system-row"><span>API</span><span className={status === 'online' ? 'ws-text-accent' : ''}>{status === 'online' ? 'Connected' : status === 'checking' || status === 'idle' ? 'Checking' : 'Offline'}</span></div>
    <div className="ws-system-row"><span>Checkpoint</span><span>{status !== 'online' ? 'Unverified' : modelLoaded ? 'Loaded' : 'Not loaded'}</span></div>
    <p>Camera starts only when you choose.</p>
  </section>
}
