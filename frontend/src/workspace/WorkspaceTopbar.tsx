import { Link } from "react-router-dom"

interface Props {
  module: string
  onMenuToggle: () => void
}

export function WorkspaceTopbar({ module, onMenuToggle }: Props) {
  return (
    <header className="ws-topbar" role="banner">
      <button
        className="ws-btn ws-btn-ghost ws-btn-sm"
        onClick={onMenuToggle}
        aria-label="Toggle sidebar"
        style={{ padding: "6px", borderRadius: "6px" }}
      >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <rect x="2" y="4" width="12" height="1.5" rx="0.75" fill="currentColor"/>
          <rect x="2" y="7.25" width="12" height="1.5" rx="0.75" fill="currentColor"/>
          <rect x="2" y="10.5" width="12" height="1.5" rx="0.75" fill="currentColor"/>
        </svg>
      </button>
      <Link to="/" className="ws-topbar-brand" aria-label="Visual Speech AI — home">
        <span className="ws-topbar-brand-mark" aria-hidden="true"/>
        VISUAL SPEECH AI
      </Link>
      <span className="ws-topbar-sep" aria-hidden="true">/</span>
      <span className="ws-topbar-module" aria-current="page">{module}</span>
      <span className="ws-topbar-spacer" />
    </header>
  )
}
