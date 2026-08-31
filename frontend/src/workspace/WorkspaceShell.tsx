import { useState, useCallback } from "react"
import { Outlet, useLocation } from "react-router-dom"
import { WorkspaceSidebar } from "./WorkspaceSidebar"
import { WorkspaceTopbar } from "./WorkspaceTopbar"
import { StatusPanel } from "./StatusPanel"

export function WorkspaceShell() {
  const [collapsed, setCollapsed] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  const toggle = useCallback(() => setCollapsed((c) => !c), [])
  const closeMobile = useCallback(() => setMobileOpen((o) => !o), [])

  const moduleMap: Record<string, string> = {
    "/demo/live": "LIVE DEMO",
    "/demo/dataset": "DATASET LAB",
    "/demo/training": "TRAINING LAB",
    "/demo/testing": "MODEL TESTING",
    "/demo/models": "MODEL REGISTRY",
    "/demo/experiments": "EXPERIMENTS",
    "/demo/pipeline": "PIPELINE",
  }
  const module = moduleMap[location.pathname] ?? "RESEARCH WORKSPACE"

  return (
    <div className="workspace" role="application" aria-label="Visual Speech AI Research Workspace">
      <WorkspaceTopbar module={module} onMenuToggle={() => setMobileOpen((o) => !o)} />
      <div className="ws-body">
        <WorkspaceSidebar
          collapsed={collapsed}
          mobileOpen={mobileOpen}
          onClose={closeMobile}
          onToggle={toggle}
        />
        <div className="ws-content" role="main" onClick={mobileOpen ? closeMobile : undefined}>
          <div className="ws-page">
            <Outlet />
          </div>
        </div>
        <StatusPanel />
      </div>
    </div>
  )
}
