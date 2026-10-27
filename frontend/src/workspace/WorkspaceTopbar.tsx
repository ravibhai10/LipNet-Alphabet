import { Link } from 'react-router-dom'
import { useBackend } from '../hooks/useBackend'
import { Icon } from './Icon'
interface Props { module: string; onMenuToggle: () => void; mobileOpen: boolean }
export function WorkspaceTopbar({ module, onMenuToggle, mobileOpen }: Props) {
  const { status, refresh } = useBackend()
  return <header className="ws-topbar">
    <button id="ws-menu-button" className="ws-menu-button ws-icon-button" onClick={onMenuToggle} aria-label="Toggle navigation" aria-expanded={mobileOpen} aria-controls="workspace-navigation"><Icon name="menu" /></button>
    <div className="ws-breadcrumb"><Link to="/demo/overview">Workspace</Link><span>/</span><strong>{module}</strong></div>
    <div className="ws-topbar-actions">
      <span className={'ws-connection ' + (status === 'online' ? 'is-online' : '')}><span className="ws-dot"/>{status === 'online' ? 'Backend connected' : status === 'checking' || status === 'idle' ? 'Checking backend' : 'Backend offline'}</span>
      <button className="ws-icon-button" onClick={refresh} disabled={status === 'checking'} aria-label="Refresh backend connection" title="Refresh backend connection"><Icon name="refresh" size={16}/></button>
      <span className="ws-topbar-divider"/><span className="ws-local-tag">LOCAL</span>
    </div>
  </header>
}
