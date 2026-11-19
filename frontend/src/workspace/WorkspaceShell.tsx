import { useState, useCallback, useEffect, useRef } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { BackendProvider, useBackend } from '../hooks/useBackend'
import { ToastProvider } from '../ui/Toast'
import { WorkspaceSidebar } from './WorkspaceSidebar'
import { WorkspaceTopbar } from './WorkspaceTopbar'
import { ExaminerDemoBar } from './components/ExaminerDemoBar'
import { QuantumParticleField } from '../visualizations/QuantumParticleField'
import { CyberCommandPalette } from './components/CyberCommandPalette'
import { CyberTelemetryHUD } from './components/CyberTelemetryHUD'

const titles: Record<string, string> = {
  '/demo/overview': 'Overview',
  '/demo/live': 'Live lip reading',
  '/demo/feedback': 'Adaptive feedback lab',
  '/demo/collect': 'Word Collection & Dataset Expansion',
  '/demo/dataset': 'Dataset explorer',
  '/demo/training': 'Training',
  '/demo/testing': 'Evaluation & testing',
  '/demo/models': 'Model registry',
  '/demo/experiments': 'Experiments',
  '/demo/pipeline': 'Pipeline',
  '/demo/docs': 'Documentation & Package',
}

export function WorkspaceShell() {
  return (
    <BackendProvider>
      <ToastProvider>
        <WorkspaceContent />
      </ToastProvider>
    </BackendProvider>
  )
}

function WorkspaceContent() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    try {
      return (localStorage.getItem('lipnet.theme') as 'dark' | 'light') || 'dark'
    } catch {
      return 'dark'
    }
  })

  const location = useLocation()
  const content = useRef<HTMLElement>(null)
  const toggle = useCallback(() => setCollapsed(c => !c), [])
  const closeMobile = useCallback(() => setMobileOpen(false), [])

  const toggleTheme = useCallback(() => {
    setTheme(prev => {
      const next = prev === 'dark' ? 'light' : 'dark'
      try {
        localStorage.setItem('lipnet.theme', next)
      } catch {
        /* Storage unavailable */
      }
      return next
    })
  }, [])

  useEffect(() => {
    setMobileOpen(false)
    content.current?.scrollTo({ top: 0 })
    content.current?.focus({ preventScroll: true })
    document.title = (titles[location.pathname] ?? 'Workspace') + ' — LipNet AI'
  }, [location.pathname])

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCommandPaletteOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', handleKey)
    return () => window.removeEventListener('keydown', handleKey)
  }, [])

  useEffect(() => {
    if (!mobileOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setMobileOpen(false)
        requestAnimationFrame(() => document.getElementById('ws-menu-button')?.focus())
      }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [mobileOpen])

  const { status, refresh } = useBackend()
  const [retryCount, setRetryCount] = useState(0)
  useEffect(() => {
    if (status === 'offline') setRetryCount(value => value + 1)
  }, [status])

  return (
      <div className={'workspace' + (collapsed ? ' ws-is-collapsed' : '') + (theme === 'light' ? ' ws-light' : '')}>
        <a href="#workspace-main" className="skip-link">Skip to workspace</a>
        {mobileOpen && <button className="ws-mobile-backdrop" onClick={closeMobile} aria-label="Close navigation" />}
        <WorkspaceSidebar collapsed={collapsed} mobileOpen={mobileOpen} onClose={closeMobile} onToggle={toggle} />
        <div className="ws-main-column" style={{ position: 'relative' }}>
          {theme === 'dark' && <QuantumParticleField particleCount={24} />}
          <WorkspaceTopbar
            module={titles[location.pathname] ?? 'Research workspace'}
            mobileOpen={mobileOpen}
            onMenuToggle={() => setMobileOpen(o => !o)}
            theme={theme}
            onToggleTheme={toggleTheme}
            onOpenCommandPalette={() => setCommandPaletteOpen(true)}
          />
          <ExaminerDemoBar />
          {status === 'offline' && <div className="ws-reconnect-banner" role="alert">
            <span>Backend disconnected. Live recognition is paused.</span>
            <strong>Retry {retryCount}</strong>
            <button type="button" onClick={refresh}>Retry now</button>
          </div>}
          <main id="workspace-main" className="ws-content" ref={content} tabIndex={-1}>
            <div className="ws-page" key={location.pathname}><Outlet /></div>
            <footer className="ws-workspace-footer">
              <span>LIPNET AI · Visual Speech Intelligence · Final Year Capstone Project (Group 12)</span>
              <span>Observe. Understand. Decode.</span>
            </footer>
          </main>
          <CyberCommandPalette
            isOpen={commandPaletteOpen}
            onClose={() => setCommandPaletteOpen(false)}
            onToggleTheme={toggleTheme}
            onRefreshBackend={refresh}
          />
          <CyberTelemetryHUD />
        </div>
      </div>
  )
}
