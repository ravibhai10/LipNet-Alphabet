import { Link } from 'react-router-dom'
import { useBackend } from '../hooks/useBackend'
import { Icon } from './Icon'

interface Props {
  module: string
  onMenuToggle: () => void
  mobileOpen: boolean
  theme: 'dark' | 'light'
  onToggleTheme: () => void
}

export function WorkspaceTopbar({ module, onMenuToggle, mobileOpen, theme, onToggleTheme }: Props) {
  const { status, refresh } = useBackend()

  return (
    <header className="ws-topbar">
      <button
        id="ws-menu-button"
        className="ws-menu-button ws-icon-button"
        onClick={onMenuToggle}
        aria-label="Toggle navigation"
        aria-expanded={mobileOpen}
        aria-controls="workspace-navigation"
      >
        <Icon name="menu" />
      </button>

      <div className="ws-breadcrumb">
        <Link to="/demo/overview">Workspace</Link>
        <span>/</span>
        <strong>{module}</strong>
      </div>

      <div className="ws-topbar-actions" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
        <span
          className="ws-badge ws-badge-ready"
          style={{ fontSize: '10px', padding: '3px 8px', letterSpacing: '0.05em' }}
          title="Final Year Capstone Project · Somaiya Trust"
        >
          GROUP 12 · CAPSTONE
        </span>

        <span className={'ws-connection ' + (status === 'online' ? 'is-online' : '')}>
          <span className="ws-dot"/>
          {status === 'online' ? 'Backend online' : status === 'checking' || status === 'idle' ? 'Checking backend' : 'Backend offline'}
        </span>

        <button
          className="ws-icon-button"
          onClick={onToggleTheme}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          style={{ cursor: 'pointer' }}
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={16} />
        </button>

        <button
          className="ws-icon-button"
          onClick={refresh}
          disabled={status === 'checking'}
          aria-label="Refresh backend connection"
          title="Refresh backend connection"
          style={{ cursor: 'pointer' }}
        >
          <Icon name="refresh" size={16} />
        </button>

        <span className="ws-topbar-divider"/>
        <span className="ws-local-tag">LOCAL WORKSPACE</span>
      </div>
    </header>
  )
}
