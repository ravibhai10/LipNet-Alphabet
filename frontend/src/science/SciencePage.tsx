import { useState, useEffect } from 'react'
import './science.css'
import { ScienceHero } from './components/ScienceHero'
import { ImageFormationScene } from './components/ImageFormationScene'
import { TemporalSequenceScene } from './components/TemporalSequenceScene'
import { ModelArchitectureScene } from './components/ModelArchitectureScene'
import { MathematicsScene } from './components/MathematicsScene'
import { PredictionAnalysisScene } from './components/PredictionAnalysisScene'
import { EvaluationScene } from './components/EvaluationScene'
import { ResearchWorkflowScene } from './components/ResearchWorkflowScene'
import { LimitationsScene } from './components/LimitationsScene'
import { ExploreResearchScene } from './components/ExploreResearchScene'

const CHAPTERS = [
  { id: 'scene-hero', label: '01 · The Question', title: 'Every Word Leaves a Trace' },
  { id: 'scene-pixels', label: '02 · Light to Pixels', title: 'Image Formation & Normalization' },
  { id: 'scene-temporal', label: '03 · Time & Motion', title: 'Articulatory Kinematics' },
  { id: 'scene-architecture', label: '04 · Neural Network', title: 'Deep Spatio-Temporal Pipeline' },
  { id: 'scene-math', label: '05 · Mathematics', title: 'Softmax, Loss & EMA Smoothing' },
  { id: 'scene-prediction', label: '06 · Prediction', title: 'Viseme Ambiguity & Shannon Entropy' },
  { id: 'scene-evaluation', label: '07 · Evaluation', title: 'Confusion Matrix & Generalization' },
  { id: 'scene-workflow', label: '08 · Reproducibility', title: 'End-to-End Scientific Lineage' },
  { id: 'scene-limitations', label: '09 · Limitations', title: 'Physical & Anatomical Boundaries' },
  { id: 'scene-explore', label: '10 · Research Labs', title: 'Next Steps & Live Demonstration' },
]

export function SciencePage() {
  const [activeChapter, setActiveChapter] = useState<string>('scene-hero')
  const [scrollProgress, setScrollProgress] = useState<number>(0)

  useEffect(() => {
    const handleScroll = () => {
      const totalScroll = document.documentElement.scrollHeight - window.innerHeight
      const currentScroll = window.scrollY
      setScrollProgress(totalScroll > 0 ? (currentScroll / totalScroll) * 100 : 0)

      // Find active section
      for (let i = CHAPTERS.length - 1; i >= 0; i--) {
        const el = document.getElementById(CHAPTERS[i].id)
        if (el) {
          const rect = el.getBoundingClientRect()
          if (rect.top <= window.innerHeight * 0.4) {
            setActiveChapter(CHAPTERS[i].id)
            break
          }
        }
      }
    }

    window.addEventListener('scroll', handleScroll, { passive: true })
    handleScroll()
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  const scrollTo = (id: string) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' })
    }
  }

  return (
    <div className="science-container">
      {/* Background celestial glow grid */}
      <div className="science-bg-grid" aria-hidden="true" />

      {/* Top Reading Progress Bar */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          height: '3px',
          width: `${scrollProgress}%`,
          background: 'linear-gradient(90deg, var(--accent, #8fdfcb), var(--info, #7de1f9))',
          zIndex: 200,
          transition: 'width 0.1s linear'
        }}
        aria-hidden="true"
      />

      {/* Floating Header */}
      <header className="science-nav">
        <a href="/" className="science-nav-brand">
          <span style={{ color: 'var(--accent)' }}>◈</span>
          <span>LIPNET-AI</span>
          <span className="science-brand-badge">Science Lab</span>
        </a>

        <div className="science-nav-links">
          <a href="/" className="science-nav-link">Main Story</a>
          <a href="/demo/overview" className="science-nav-link">Workspace</a>
          <a href="/demo/live" className="science-nav-cta"><span className="science-nav-launch">Launch </span>Live Demo</a>
        </div>
      </header>

      {/* Floating Chapter Progress Indicator */}
      <nav className="science-chapter-bar" aria-label="Story Chapters">
        {CHAPTERS.map((chap) => {
          const isActive = activeChapter === chap.id
          return (
            <button
              key={chap.id}
              onClick={() => scrollTo(chap.id)}
              className={`science-chapter-dot ${isActive ? 'active' : ''}`}
              aria-label={`Jump to ${chap.label}: ${chap.title}`}
            >
              <span className="science-chapter-tooltip">
                <strong>{chap.label}</strong> — {chap.title}
              </span>
            </button>
          )
        })}
      </nav>

      {/* Main Scientific Narrative Container */}
      <main className="science-main" id="main">
        <ScienceHero />
        <ImageFormationScene />
        <TemporalSequenceScene />
        <ModelArchitectureScene />
        <MathematicsScene />
        <PredictionAnalysisScene />
        <EvaluationScene />
        <ResearchWorkflowScene />
        <LimitationsScene />
        <ExploreResearchScene />
      </main>
    </div>
  )
}
