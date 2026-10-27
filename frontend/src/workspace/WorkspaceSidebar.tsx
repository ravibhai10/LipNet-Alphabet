import { Link, NavLink } from 'react-router-dom'
import { useEffect, useRef } from 'react'
import { BrandMark, Icon } from './Icon'
import { StatusPanel } from './StatusPanel'

interface Props { collapsed: boolean; mobileOpen: boolean; onClose: () => void; onToggle: () => void }
const groups: { label: string; items: { path: string; href?: string; label: string; icon: string }[] }[] = [
  { label: 'WORKSPACE', items: [
    { path: 'overview', label: 'Overview', icon: 'grid' },
    { path: 'live', label: 'Live lip reading', icon: 'camera' },
  ]},
  { label: 'RESEARCH', items: [
    { path: 'dataset', label: 'Dataset explorer', icon: 'database' },
    { path: 'training', label: 'Training', icon: 'activity' },
    { path: 'testing', label: 'Evaluation & testing', icon: 'chart' },
    { path: 'models', label: 'Model registry', icon: 'layers' },
    { path: 'experiments', label: 'Experiments', icon: 'flask' },
  ]},
  { label: 'SYSTEM', items: [
    { path: 'science', href: '/science', label: 'Science Lab', icon: 'flask' },
    { path: 'research', href: '/research', label: 'Animated research', icon: 'activity' },
    { path: 'pipeline', label: 'Pipeline', icon: 'pipeline' },
    { path: 'docs', label: 'Documentation', icon: 'book' },
  ]},
]
export function WorkspaceSidebar({ collapsed, mobileOpen, onClose, onToggle }: Props) {
  const sidebarRef = useRef<HTMLElement>(null)
  useEffect(() => {
    const media = window.matchMedia('(max-width: 760px)')
    const update = () => {
      if (!sidebarRef.current) return
      sidebarRef.current.inert = media.matches && !mobileOpen
      const main = document.querySelector<HTMLElement>('.ws-main-column')
      if (main) main.inert = media.matches && mobileOpen
      if (media.matches && !mobileOpen) sidebarRef.current.setAttribute('aria-hidden', 'true')
      else sidebarRef.current.removeAttribute('aria-hidden')
    }
    update()
    if (media.matches && mobileOpen) sidebarRef.current?.querySelector<HTMLAnchorElement>('.ws-nav-item.active')?.focus()
    media.addEventListener('change', update)
    return () => { media.removeEventListener('change', update); const main = document.querySelector<HTMLElement>('.ws-main-column'); if (main) main.inert = false }
  }, [mobileOpen])
  return (
    <aside ref={sidebarRef} id="workspace-navigation" className={'ws-sidebar' + (collapsed ? ' collapsed' : '') + (mobileOpen ? ' mobile-open' : '')}>
      <Link className="ws-sidebar-brand" to="/" aria-label="LipNet AI home">
        <BrandMark /><span className="ws-brand-copy"><strong>LIPNET<span>AI</span></strong><small>VISUAL SPEECH INTELLIGENCE</small></span>
      </Link>
      <div className="ws-workspace-label"><span className="ws-workspace-avatar">R</span><span>Research workspace<small>Local environment</small></span><Icon name="chevron" size={14} /></div>
      <nav className="ws-sidebar-nav" aria-label="Workspace navigation">
        {groups.map(group => <div className="ws-nav-group" key={group.label}>
          <p className="ws-sidebar-section-label">{group.label}</p>
          {group.items.map(item => <NavLink key={item.path} to={item.href ?? '/demo/' + item.path} onClick={onClose} title={collapsed ? item.label : undefined} aria-label={item.label} className={({ isActive }) => 'ws-nav-item' + (isActive ? ' active' : '')}>
            <Icon name={item.icon} /><span className="ws-nav-label">{item.label}</span>{item.path === 'live' && <span className="ws-nav-small">LIVE</span>}
          </NavLink>)}
        </div>)}
      </nav>
      <StatusPanel />
      <div className="ws-sidebar-bottom"><Link to="/" className="ws-home-link" title="Back to website"><Icon name="arrow" size={16}/><span>Back to website</span></Link><button className="ws-sidebar-toggle" onClick={onToggle} aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'} aria-expanded={!collapsed}><Icon name="collapse" size={17}/></button></div>
    </aside>
  )
}
