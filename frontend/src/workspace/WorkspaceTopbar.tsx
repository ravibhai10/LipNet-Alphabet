import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { useBackend } from '../hooks/useBackend'
import { cyberAudio } from '../utils/CyberSoundFX'
import { useToast } from '../ui/Toast'
import { Icon } from './Icon'

interface Props {
  module: string
  onMenuToggle: () => void
  mobileOpen: boolean
  theme: 'dark' | 'light'
  onToggleTheme: () => void
  onOpenCommandPalette?: () => void
}

export function WorkspaceTopbar({ module, onMenuToggle, mobileOpen, theme, onToggleTheme, onOpenCommandPalette }: Props) {
  const { status, refresh } = useBackend()
  const { info } = useToast()
  const [time, setTime] = useState(() => new Date().toLocaleTimeString())
  const [soundOn, setSoundOn] = useState(() => cyberAudio.isEnabled())

  useEffect(() => {
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000)
    return () => clearInterval(timer)
  }, [])

  const handleSoundToggle = () => {
    const next = cyberAudio.toggle()
    setSoundOn(next)
    info(next ? 'Sound effects enabled' : 'Sound effects muted', 'Audio')
  }

  const handleThemeToggle = () => {
    onToggleTheme()
    info(`Switched to ${theme === 'dark' ? 'light' : 'dark'} mode`, 'Theme')
  }

  const handleRefresh = () => {
    refresh()
    info('Pinging backend at 127.0.0.1:8001...', 'Connection')
  }

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

      <div className="ws-topbar-actions" style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: 'var(--ws-text-muted)',
          background: 'var(--ws-surface2)',
          border: '1px solid var(--ws-border)',
          borderRadius: '6px',
          padding: '4px 8px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}>
          <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: 'var(--ws-accent)' }} />
          <span style={{ fontVariantNumeric: 'tabular-nums' }}>{time}</span>
        </div>

        <button
          className="ws-icon-button"
          onClick={() => {
            cyberAudio.playClick()
            if (onOpenCommandPalette) onOpenCommandPalette()
            else window.dispatchEvent(new KeyboardEvent('keydown', { key: 'k', ctrlKey: true }))
          }}
          title="Search modules & commands (Ctrl+K)"
          aria-label="Search modules and commands"
          style={{
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            width: 'auto',
            padding: '0 8px',
            fontSize: '11px',
            fontFamily: 'var(--font-mono)',
            background: 'var(--ws-surface2)',
            border: '1px solid var(--ws-border)',
            borderRadius: '6px'
          }}
        >
          <Icon name="search" size={13} />
          <span style={{ opacity: 0.7 }}>Ctrl+K</span>
        </button>

        <button
          className="ws-icon-button"
          onClick={handleSoundToggle}
          aria-label={soundOn ? 'Mute sound effects' : 'Enable sound effects'}
          title={soundOn ? 'Mute sound effects' : 'Enable sound effects'}
          style={{ cursor: 'pointer', color: soundOn ? 'var(--ws-accent)' : 'var(--ws-text-dim)' }}
        >
          <Icon name={soundOn ? 'volume' : 'volume-x'} size={15} />
        </button>

        <span
          className="ws-badge ws-badge-ready"
          style={{ fontSize: '10px', padding: '3px 8px', letterSpacing: '0.04em' }}
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
          onClick={handleThemeToggle}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          style={{ cursor: 'pointer' }}
        >
          <Icon name={theme === 'dark' ? 'sun' : 'moon'} size={15} />
        </button>

        <button
          className="ws-icon-button"
          onClick={handleRefresh}
          disabled={status === 'checking'}
          aria-label="Refresh backend connection"
          title="Refresh backend connection"
          style={{ cursor: 'pointer' }}
        >
          <Icon name="refresh" size={15} />
        </button>

        <span className="ws-topbar-divider"/>
        <span className="ws-local-tag">LOCAL WORKSPACE</span>
      </div>
    </header>
  )
}
