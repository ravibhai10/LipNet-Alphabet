import { BackendProvider } from "./hooks/useBackend"
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
import { FaceScanner } from "./sections/FaceScanner"
import { ConformerExplorer } from "./sections/ConformerExplorer"
import { PredictionPlayground } from "./sections/PredictionPlayground"
import { GlobalMap } from "./sections/GlobalMap"
import { AblationStudio } from "./sections/AblationStudio"

export function App() {
  return (
    <BackendProvider>
      <a className="skip-link" href="#main">Skip to content</a>
      <Nav />
      <main id="main">
        <Hero />
        <FaceScanner />
        <Phenomenon />
        <Pipeline />
        <Streams />
        <Fusion />
        <ConformerExplorer />
        <TimeMatters />
        <TemporalEngine />
        <Decoding />
        <Uncertainty />
        <LiveDemoSection />
        <PredictionPlayground />
        <ResearchPipeline />
        <GlobalMap />
        <AblationStudio />
        <DataSection />
        <Results />
        <Architecture />
        <FinalStatement />
      </main>
      <Footer />
    </BackendProvider>
  )
}
