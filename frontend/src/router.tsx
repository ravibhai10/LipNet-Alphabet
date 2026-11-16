/**
 * Router configuration with dynamic chunk fallback and error boundaries.
 */
import { createBrowserRouter, Navigate, useRouteError } from 'react-router-dom'
import { lazy, Suspense } from 'react'
import { WorkspaceShell } from './workspace/WorkspaceShell'
import { LiveDemoPage } from './workspace/pages/LiveDemo'
import { RecordingStudioPage } from './workspace/pages/RecordingStudio'
import { DatasetPage } from './workspace/pages/Dataset'
import { TrainingPage } from './workspace/pages/Training'
import { TestingPage } from './workspace/pages/Testing'
import { ModelsPage } from './workspace/pages/Models'
import { ExperimentsPage } from './workspace/pages/Experiments'
import { PipelinePage } from './workspace/pages/Pipeline'
import { OverviewPage } from './workspace/pages/Overview'
import { DocumentationPage } from './workspace/pages/Documentation'

function RouteErrorBoundary() {
  const error = useRouteError() as any
  const msg = error?.message || error?.toString?.() || ''
  const isChunkError =
    msg.includes('dynamically imported module') ||
    msg.includes('Failed to fetch') ||
    msg.includes('chunk')

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      backgroundColor: '#0a0d14',
      color: '#f8fafc',
      fontFamily: 'system-ui, -apple-system, sans-serif',
      padding: '2rem'
    }}>
      <div style={{
        maxWidth: '520px',
        width: '100%',
        backgroundColor: '#111827',
        border: '1px solid #1f2937',
        borderRadius: '16px',
        padding: '2.5rem',
        textAlign: 'center',
        boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.7)'
      }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>⚡</div>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: '0.75rem', color: '#38bdf8' }}>
          {isChunkError ? 'Application Update Available' : 'Application Navigation Notice'}
        </h2>
        <p style={{ color: '#94a3b8', fontSize: '0.92rem', lineHeight: '1.6', marginBottom: '1.75rem' }}>
          {isChunkError
            ? 'A fresh build of LipNet-AI has been compiled. Please refresh to load the latest application assets.'
            : (error?.message || 'An unexpected error occurred while loading this page.')}
        </p>
        <button
          onClick={() => {
            sessionStorage.clear()
            window.location.reload()
          }}
          style={{
            backgroundColor: '#0284c7',
            color: '#ffffff',
            border: 'none',
            borderRadius: '8px',
            padding: '0.75rem 1.75rem',
            fontSize: '0.92rem',
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'background-color 0.15s ease',
          }}
          onMouseOver={(e) => (e.currentTarget.style.backgroundColor = '#0369a1')}
          onMouseOut={(e) => (e.currentTarget.style.backgroundColor = '#0284c7')}
        >
          Reload State
        </button>
      </div>
    </div>
  )
}

const App = lazy(() => import('./App').then(module => ({ default: module.App })))
const ResearchApp = lazy(() => import('./ResearchApp').then(module => ({ default: module.App })))
const SciencePage = lazy(() => import('./science/SciencePage').then(module => ({ default: module.SciencePage })))

export const router = createBrowserRouter([
  {
    path: '/science',
    errorElement: <RouteErrorBoundary />,
    element: <Suspense fallback={<p className="mono-note" style={{ padding: '3rem' }}>Loading Science Lab…</p>}><SciencePage /></Suspense>
  },
  {
    path: '/research',
    errorElement: <RouteErrorBoundary />,
    element: <Suspense fallback={<p className="mono-note" style={{padding:'3rem'}}>Loading research exhibits…</p>}><ResearchApp /></Suspense>
  },
  {
    path: '/',
    errorElement: <RouteErrorBoundary />,
    element: <Suspense fallback={<p className="mono-note" style={{ padding: '3rem' }}>Loading Visual Speech AI…</p>}><App /></Suspense>,
  },
  {
    path: '/demo',
    errorElement: <RouteErrorBoundary />,
    element: <WorkspaceShell />,
    children: [
      { index: true, element: <Navigate to="overview" replace /> },
      { path: 'overview', element: <OverviewPage /> },
      { path: 'docs', element: <DocumentationPage /> },
      { path: 'live', element: <LiveDemoPage /> },
      { path: 'collect', element: <RecordingStudioPage /> },
      { path: 'dataset', element: <DatasetPage /> },
      { path: 'training', element: <TrainingPage /> },
      { path: 'testing', element: <TestingPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'experiments', element: <ExperimentsPage /> },
      { path: 'pipeline', element: <PipelinePage /> },
    ],
  },
])
