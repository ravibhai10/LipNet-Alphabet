#!/usr/bin/env python3
"""Build workspace files using direct write to avoid editor tool constraints."""
import os

ROOT = r"c:\Users\ravik\OneDrive\Desktop\lip alphabet\frontend\src"

def w(path, content):
    full = os.path.join(ROOT, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created {path} ({len(content)} bytes)")

# ── 1. main.tsx ──────────────────────────────────────────────────────────────
w("main.tsx", '''import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { RouterProvider } from "react-router-dom"
import { router } from "./router"
import "./styles/global.css"
import "./styles/nav-hero.css"
import "./styles/sections.css"
import "./styles/sections2.css"
import "./styles/workspace.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
''')

# ── 2. router.tsx (already created, skip) ─────────────────────────────────────

# ── 3. App.tsx ────────────────────────────────────────────────────────────────
w("App.tsx", '''import { BackendProvider } from "./hooks/useBackend"
import { Nav } from "./components/Nav"
import { Hero } from "./sections/Hero"
import { Phenomenon } from "./sections/Phenomenon"
import { Pipeline } from "./sections/Pipeline"
import { Streams } from "./sections/Streams"
import { Fusion } from "./sections/Fusion"
import { TimeMatters } from "./sections/TimeMatters"
import { TemporalEngine } from "./sections/TemporalEngine"
import { Decoding } from "./sections/Decoding"
import { Uncertainty } from "./sections/Uncertainty"
import { LiveDemoSection } from "./sections/LiveDemoSection"
import { ResearchPipeline } from "./sections/ResearchPipeline"
import { DataSection } from "./sections/DataSection"
import { Results } from "./sections/Results"
import { Architecture } from "./sections/Architecture"
import { FinalStatement } from "./sections/FinalStatement"
import { Footer } from "./components/Footer"

export function App() {
  return (
    <BackendProvider>
      <a className="skip-link" href="#main">Skip to content</a>
      <Nav />
      <main id="main">
        <Hero />
        <Phenomenon />
        <Pipeline />
        <Streams />
        <Fusion />
        <TimeMatters />
        <TemporalEngine />
        <Decoding />
        <Uncertainty />
        <LiveDemoSection />
        <ResearchPipeline />
        <DataSection />
        <Results />
        <Architecture />
        <FinalStatement />
      </main>
      <Footer />
    </BackendProvider>
  )
}
''')

print("\\nDone writing entry files.")
