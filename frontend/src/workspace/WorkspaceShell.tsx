import { useState, useCallback, useEffect, useRef } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { BackendProvider } from '../hooks/useBackend'
import { WorkspaceSidebar } from './WorkspaceSidebar'
import { WorkspaceTopbar } from './WorkspaceTopbar'

const titles: Record<string, string> = {
  '/demo/overview': 'Overview', '/demo/live': 'Live lip reading',
  '/demo/dataset': 'Dataset explorer', '/demo/training': 'Training',
  '/demo/testing': 'Evaluation & testing', '/demo/models': 'Model registry',
  '/demo/experiments': 'Experiments', '/demo/pipeline': 'Pipeline',
  '/demo/docs': 'Documentation',
}
export function WorkspaceShell() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()
  const content = useRef<HTMLElement>(null)
  const toggle = useCallback(() => setCollapsed(c => !c), [])
  const closeMobile = useCallback(() => setMobileOpen(false), [])
  useEffect(() => {
    setMobileOpen(false)
    content.current?.scrollTo({ top: 0 })
    content.current?.focus({ preventScroll: true })
    document.title = (titles[location.pathname] ?? 'Workspace') + ' — LipNet AI'
  }, [location.pathname])
  useEffect(() => {
    if (!mobileOpen) return
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setMobileOpen(false); requestAnimationFrame(() => document.getElementById('ws-menu-button')?.focus()) }
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [mobileOpen])
  return (
    <BackendProvider>
      <div className={'workspace' + (collapsed ? ' ws-is-collapsed' : '')}>
        <a href="#workspace-main" className="skip-link">Skip to workspace</a>
        {mobileOpen && <button className="ws-mobile-backdrop" onClick={closeMobile} aria-label="Close navigation" />}
        <WorkspaceSidebar collapsed={collapsed} mobileOpen={mobileOpen} onClose={closeMobile} onToggle={toggle} />
        <div className="ws-main-column">
          <WorkspaceTopbar module={titles[location.pathname] ?? 'Research workspace'} mobileOpen={mobileOpen} onMenuToggle={() => setMobileOpen(o => !o)} />
          <main id="workspace-main" className="ws-content" ref={content} tabIndex={-1}>
            <div className="ws-page" key={location.pathname}><Outlet /></div>
            <footer className="ws-workspace-footer"><span>LIPNET AI <i /> Visual speech research</span><span>Observe. Understand. Decode.</span></footer>
          </main>
        </div>
      </div>
    </BackendProvider>
  )
}
