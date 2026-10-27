/**
 * Router configuration.
 *
 * /              → six-scene story and simple live demonstration
 * /research      → preserved detailed research exhibits
 * /demo          → research workspace shell
 *   /demo/live        → Live Demo
 *   /demo/dataset     → Dataset Lab
 *   /demo/training    → Training Lab
 *   /demo/testing     → Testing Lab
 *   /demo/models      → Model Registry
 *   /demo/experiments → Experiments
 *   /demo/pipeline    → Architecture Pipeline
 */
import { createBrowserRouter, Navigate, Outlet, useLocation } from 'react-router-dom'
import { lazy, Suspense, useLayoutEffect } from 'react'
import { WorkspaceShell } from './workspace/WorkspaceShell'
import { LiveDemoPage } from './workspace/pages/LiveDemo'
import { DatasetPage } from './workspace/pages/Dataset'
import { TrainingPage } from './workspace/pages/Training'
import { TestingPage } from './workspace/pages/Testing'
import { ModelsPage } from './workspace/pages/Models'
import { ExperimentsPage } from './workspace/pages/Experiments'
import { PipelinePage } from './workspace/pages/Pipeline'
import { OverviewPage } from './workspace/pages/Overview'
import { DocumentationPage } from './workspace/pages/Documentation'

const App = lazy(() => import('./App').then(module => ({ default: module.App })))
const ResearchApp = lazy(() => import('./ResearchApp').then(module => ({ default: module.App })))
const SciencePage = lazy(() => import('./science/SciencePage').then(module => ({ default: module.SciencePage })))

function ProjectRouteLayout() {
  const { pathname, hash } = useLocation()
  useLayoutEffect(() => {
    if (!hash) { window.scrollTo({ top: 0, left: 0, behavior: 'instant' }); return }
    let frame = 0, attempts = 0
    const findAnchor = () => {
      const target = document.getElementById(hash.slice(1))
      if (target) target.scrollIntoView({ behavior: 'instant', block: 'start' })
      else if (attempts++ < 120) frame = requestAnimationFrame(findAnchor)
    }
    findAnchor()
    return () => cancelAnimationFrame(frame)
    // Same-page anchor links retain the browser's smooth scrolling.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname])
  return <Outlet />
}

export const router = createBrowserRouter([{ element: <ProjectRouteLayout />, children: [
  { path: '/science', element: <Suspense fallback={<p className="mono-note" style={{ padding: '3rem' }}>Loading Science Lab…</p>}><SciencePage /></Suspense> },
  { path: '/research', element: <Suspense fallback={<p className="mono-note" style={{padding:'3rem'}}>Loading Visual Speech AI Story…</p>}><App /></Suspense> },
  {
    path: '/',
    element: <Suspense fallback={<p className="mono-note" style={{ padding: '3rem' }}>Loading Visual Speech AI…</p>}><ResearchApp /></Suspense>,
  },
  {
    path: '/demo',
    element: <WorkspaceShell />,
    children: [
      { index: true, element: <Navigate to="overview" replace /> },
      { path: 'overview', element: <OverviewPage /> },
      { path: 'docs', element: <DocumentationPage /> },
      { path: 'live', element: <LiveDemoPage /> },
      { path: 'dataset', element: <DatasetPage /> },
      { path: 'training', element: <TrainingPage /> },
      { path: 'testing', element: <TestingPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'experiments', element: <ExperimentsPage /> },
      { path: 'pipeline', element: <PipelinePage /> },
    ],
  },
]}])
