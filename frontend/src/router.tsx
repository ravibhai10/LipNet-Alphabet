/**
 * Router configuration.
 *
 * /              → existing landing-page SPA (all sections)
 * /demo          → research workspace shell
 *   /demo/live        → Live Demo
 *   /demo/dataset     → Dataset Lab
 *   /demo/training    → Training Lab
 *   /demo/testing     → Testing Lab
 *   /demo/models      → Model Registry
 *   /demo/experiments → Experiments
 *   /demo/pipeline    → Architecture Pipeline
 */
import { createBrowserRouter, Navigate } from 'react-router-dom'
import { App } from './App'
import { WorkspaceShell } from './workspace/WorkspaceShell'
import { LiveDemoPage } from './workspace/pages/LiveDemo'
import { DatasetPage } from './workspace/pages/Dataset'
import { TrainingPage } from './workspace/pages/Training'
import { TestingPage } from './workspace/pages/Testing'
import { ModelsPage } from './workspace/pages/Models'
import { ExperimentsPage } from './workspace/pages/Experiments'
import { PipelinePage } from './workspace/pages/Pipeline'

export const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
  },
  {
    path: '/demo',
    element: <WorkspaceShell />,
    children: [
      { index: true, element: <Navigate to="live" replace /> },
      { path: 'live', element: <LiveDemoPage /> },
      { path: 'dataset', element: <DatasetPage /> },
      { path: 'training', element: <TrainingPage /> },
      { path: 'testing', element: <TestingPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'experiments', element: <ExperimentsPage /> },
      { path: 'pipeline', element: <PipelinePage /> },
    ],
  },
])
