import { NavLink } from "react-router-dom"

interface Props {
  collapsed: boolean
  mobileOpen: boolean
  onClose: () => void
  onToggle: () => void
}

const NAV_ITEMS = [
  { path: "/demo/live",        label: "Live Demo",        section: "ANALYZE" },
  { path: "/demo/dataset",     label: "Dataset",          section: "DATA" },
  { path: "/demo/training",    label: "Training",         section: "TRAIN" },
  { path: "/demo/testing",      label: "Testing",          section: "TEST" },
  { path: "/demo/models",       label: "Models",           section: "MANAGE" },
  { path: "/demo/experiments",  label: "Experiments",      section: "MANAGE" },
  { path: "/demo/pipeline",     label: "Pipeline",          section: "DOCS" },
]

const ICON_PATHS: Record<string, string> = {
  live:        '<circle cx="9" cy="9" r="3" fill="currentColor"/><path d="M9 2C5.13 2 2 5.13 2 9s3.13 7 7 7 7-3.13 7-7-3.13-7-7-7z" stroke="currentColor" stroke-width="1.5"/>',
  dataset:     '<rect x="2" y="3" width="14" height="3" rx="1" fill="currentColor" opacity=".6"/><rect x="2" y="7.5" width="14" height="3" rx="1" fill="currentColor" opacity=".8"/><rect x="2" y="12" width="14" height="3" rx="1" fill="currentColor"/>',
  training:    '<path d="M3 14L7 9l3 3 5-6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
  testing:     '<path d="M9 2v4M9 12v4M2 9h4M12 9h4M4.22 4.22l2.83 2.83M10.95 10.95l2.83 2.83M13.78 4.22l-2.83 2.83M7.05 10.95l-2.83 2.83" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/><circle cx="9" cy="9" r="3" fill="currentColor"/>',
  models:      '<rect x="2" y="6" width="14" height="8" rx="1.5" stroke="currentColor" stroke-width="1.5"/><path d="M6 6V4a1 1 0 011-1h4a1 1 0 011 1v2" stroke="currentColor" stroke-width="1.5"/><circle cx="9" cy="10" r="1.5" fill="currentColor"/>',
  experiments: '<path d="M6 2v5L2 14h14L12 7V2H6z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/><path d="M7 14v2M11 14v2" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
  pipeline:    '<path d="M3 9h4M11 9h4M7 9a2 2 0 100-4 2 2 0 000 4zM11 9a2 2 0 100 4 2 2 0 000-4z" stroke="currentColor" stroke-width="1.5"/>',
  settings:    '<circle cx="9" cy="9" r="2.5" stroke="currentColor" stroke-width="1.5"/><path d="M9 2v1.5M9 14.5V16M2 9h1.5M14.5 9H16M3.76 3.76l1.06 1.06M13.18 13.18l1.06 1.06M14.24 3.76l-1.06 1.06M4.82 13.18l-1.06 1.06" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
}

function NavIcon({ type }: { type: string }) {
  const d = ICON_PATHS[type]
  if (!d) return null
  return (
    <svg viewBox="0 0 18 18" fill="none" className="ws-nav-icon" aria-hidden="true" dangerouslySetInnerHTML={{ __html: d }} />
  )
}

export function WorkspaceSidebar({ collapsed, mobileOpen, onClose, onToggle }: Props) {
  const sidebarClass = ["ws-sidebar", collapsed ? "collapsed" : "", mobileOpen ? "mobile-open" : ""].filter(Boolean).join(" ")
  const sections = ["ANALYZE","DATA","TRAIN","TEST","MANAGE","DOCS"] as const

  return (
    <nav className={sidebarClass} aria-label="Workspace navigation">
      <div className="ws-sidebar-brand" aria-hidden="true">
        <span className="ws-sidebar-brand-mark"/>
        {!collapsed && <span>RESEARCH WORKSPACE</span>}
      </div>

      <div className="ws-sidebar-nav">
        {sections.map((section) => {
          const items = NAV_ITEMS.filter((i) => i.section === section)
          return (
            <div key={section}>
              {!collapsed && <div className="ws-sidebar-section-label">{section}</div>}
              {items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) => ["ws-nav-item", isActive ? "active" : ""].filter(Boolean).join(" ")}
                  onClick={onClose}
                  title={collapsed ? item.label : undefined}
                >
                  <NavIcon type={item.path.replace("/demo/", "")} />
                  <span className="ws-nav-label">{item.label}</span>
                </NavLink>
              ))}
            </div>
          )
        })}
      </div>

      <div className="ws-sidebar-bottom">
        <a className="ws-nav-item bottom" href="/" title={collapsed ? "Back to Website" : undefined}>
          <NavIcon type="settings" />
          <span className="ws-nav-label">Back to Website</span>
        </a>
      </div>

      <button className="ws-sidebar-toggle" onClick={onToggle} aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
        <svg className="ws-toggle-icon" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M10 3L5 8l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
    </nav>
  )
}
