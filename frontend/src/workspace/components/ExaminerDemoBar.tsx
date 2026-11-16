import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Icon } from '../Icon'

export interface DemoStep {
  id: number
  title: string
  path: string
  mode?: string
  timeEstimate: string
  examinerFocus: string
  talkingPoint: string
}

export const EXAMINER_STEPS: DemoStep[] = [
  {
    id: 1,
    title: 'Project Overview',
    path: '/demo/overview',
    timeEstimate: '1.5 min',
    examinerFocus: 'Research Scope & Viseme Ambiguity',
    talkingPoint: 'Similar visible mouth shapes limit isolated letters. The preserved eight-word model scored 182/192 on historical test recordings; physical webcam accuracy is unproven.'
  },
  {
    id: 2,
    title: 'Real-Time Vision Pipeline',
    path: '/demo/live',
    mode: 'alphabet',
    timeEstimate: '1.5 min',
    examinerFocus: 'MediaPipe Landmarks & 96px Mouth Bounding',
    talkingPoint: 'Vision-only silent speech. On-device 478 landmarks, eye-axis roll angle compensation, aperture ratio.'
  },
  {
    id: 3,
    title: 'Eight-Word Recognition',
    path: '/demo/live',
    mode: 'word',
    timeEstimate: '2.0 min',
    examinerFocus: 'AV-HuBERT 103M Encoder + Layer-9 Head',
    talkingPoint: 'Capture one supported word, such as hello. Historical top-1 was 182/192 under the recorded protocol, which includes prior test-speaker exposure.'
  },
  {
    id: 4,
    title: 'Abstention & Rejection',
    path: '/demo/live',
    mode: 'word',
    timeEstimate: '1.0 min',
    examinerFocus: 'Active Rejection Policy (0.95 / 0.20 Gate)',
    talkingPoint: 'The score gate and temporal agreement can reject uncertain clips. Some phrases still pass; this is not validated detection of unseen words.'
  },
  {
    id: 5,
    title: 'Personal Word Adaptation',
    path: '/demo/collect',
    timeEstimate: '2.0 min',
    examinerFocus: 'Reviewed Data Collection and Controlled Batch Training',
    talkingPoint: 'HELP can be collected as a separate target. It does not expand the active eight-word head. Reviewed data, a frozen protocol and paired evaluation precede any explicit deployment.'
  },
  {
    id: 6,
    title: 'Dataset Management',
    path: '/demo/dataset',
    timeEstimate: '1.0 min',
    examinerFocus: 'GRID (32.8k) vs MIRACL (1.2k) vs Personal',
    talkingPoint: 'Verifiable multi-speaker splits. Strict partition isolation between train, validation, and held-out test sets.'
  },
  {
    id: 7,
    title: 'Benchmark Evidence',
    path: '/demo/testing',
    timeEstimate: '1.5 min',
    examinerFocus: 'Held-Out Confusion Matrix & Error Tradeoffs',
    talkingPoint: '182/192 correct on unseen test speakers. Transparent confusion analysis between alveolar/bilabial plosives.'
  },
  {
    id: 8,
    title: 'Neural Architecture',
    path: '/demo/pipeline',
    timeEstimate: '1.0 min',
    examinerFocus: 'Dual-Track System Architecture & Hardware CPU',
    talkingPoint: 'Strict isolation of Alphabet Conv2D-BiLSTM and Whole-Word AV-HuBERT branches running on standard CPU.'
  }
]

export function ExaminerDemoBar() {
  const navigate = useNavigate()
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  const [activeStepIndex, setActiveStepIndex] = useState(() => {
    const idx = EXAMINER_STEPS.findIndex(s => s.path === location.pathname)
    return idx >= 0 ? idx : 0
  })

  // Synchronize stepper if location changes
  useEffect(() => {
    const idx = EXAMINER_STEPS.findIndex(s => s.path === location.pathname)
    if (idx >= 0 && idx !== activeStepIndex) {
      setActiveStepIndex(idx)
    }
  }, [location.pathname])

  const step = EXAMINER_STEPS[activeStepIndex] || EXAMINER_STEPS[0]

  const goToStep = (index: number) => {
    if (index < 0 || index >= EXAMINER_STEPS.length) return
    setActiveStepIndex(index)
    navigate(EXAMINER_STEPS[index].path)
  }

  return (
    <aside
      className={`ws-examiner-hud ${collapsed ? 'is-collapsed' : ''}`}
      aria-label="External Examiner Demonstration Navigator"
    >
      <div className="ws-examiner-bar">
        <div className="ws-examiner-left">
          <div className="ws-examiner-badge">
            <span className="ws-examiner-live-dot" />
            <strong>EXAMINER DEMO MODE</strong>
            <span className="ws-examiner-step-pill">
              STEP {step.id} / {EXAMINER_STEPS.length}
            </span>
            <span className="ws-examiner-time">{step.timeEstimate}</span>
          </div>

          {!collapsed && (
            <div className="ws-examiner-content">
              <div className="ws-examiner-title-row">
                <span className="ws-examiner-step-name">{step.title}</span>
                <span className="ws-examiner-focus">🎯 Focus: {step.examinerFocus}</span>
              </div>
              <p className="ws-examiner-talk">
                <strong>Presenter Cue:</strong> {step.talkingPoint}
              </p>
            </div>
          )}
        </div>

        <div className="ws-examiner-controls">
          <button
            type="button"
            className="ws-btn ws-btn-ghost ws-btn-sm ws-examiner-nav-btn"
            disabled={activeStepIndex === 0}
            onClick={() => goToStep(activeStepIndex - 1)}
            title="Previous demonstration step"
          >
            ← Prev
          </button>

          <div className="ws-examiner-dots">
            {EXAMINER_STEPS.map((s, idx) => (
              <button
                key={s.id}
                type="button"
                className={`ws-examiner-dot ${idx === activeStepIndex ? 'is-active' : ''}`}
                onClick={() => goToStep(idx)}
                title={`Step ${s.id}: ${s.title}`}
                aria-label={`Step ${s.id}: ${s.title}`}
              />
            ))}
          </div>

          <button
            type="button"
            className="ws-btn ws-btn-primary ws-btn-sm ws-examiner-nav-btn"
            disabled={activeStepIndex === EXAMINER_STEPS.length - 1}
            onClick={() => goToStep(activeStepIndex + 1)}
            title="Next demonstration step"
          >
            Next Step →
          </button>

          <button
            type="button"
            className="ws-icon-button ws-examiner-toggle"
            onClick={() => setCollapsed(c => !c)}
            title={collapsed ? 'Expand Examiner Demonstration HUD' : 'Collapse HUD'}
            aria-label={collapsed ? 'Expand Examiner Demonstration HUD' : 'Collapse HUD'}
          >
            <Icon name={collapsed ? 'chevron' : 'collapse'} size={14} />
          </button>
        </div>
      </div>
    </aside>
  )
}
