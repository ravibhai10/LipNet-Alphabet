import { useState, useEffect } from 'react'
import { cyberAudio } from '../../utils/CyberSoundFX'

interface QuickStartModalProps {
  isOpen: boolean
  onClose: () => void
}

export function QuickStartModal({ isOpen, onClose }: QuickStartModalProps) {
  const [step, setStep] = useState(1)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return
      if (e.key === 'Escape') {
        onClose()
      } else if (e.key === 'ArrowRight' && step < 3) {
        cyberAudio.playClick()
        setStep(s => s + 1)
      } else if (e.key === 'ArrowLeft' && step > 1) {
        cyberAudio.playClick()
        setStep(s => s - 1)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, step, onClose])

  if (!isOpen) return null

  const handleNext = () => {
    cyberAudio.playClick()
    if (step < 3) {
      setStep(s => s + 1)
    } else {
      try {
        localStorage.setItem('lipnet.quickstart_seen', 'true')
      } catch {
        /* storage unavailable */
      }
      onClose()
    }
  }

  const handlePrev = () => {
    cyberAudio.playClick()
    if (step > 1) setStep(s => s - 1)
  }

  const steps = [
    {
      id: 1,
      badge: 'STEP 01 / OPTICAL LOCK',
      title: 'Align Face & Check Pre-Flight Gate',
      desc: 'Position your face inside the targeting reticle. Ensure your mouth is clearly visible. The Pre-Flight Telemetry automatically verifies Lighting, Centering, Distance, and 30 Hz Stability.',
      hotkey: 'Space',
      hotkeyAction: 'Start / Stop Camera Feed',
      icon: 'camera',
    },
    {
      id: 2,
      badge: 'STEP 02 / ARTICULATION',
      title: 'Deliberate Word Articulation',
      desc: 'Visual-speech is vision-only (microphone is disabled by design). Click "Start Word" (or say a target word like HELLO, HELP, STOP). The dynamic MAR oscilloscope tracks lip opening in real time.',
      hotkey: 'V',
      hotkeyAction: 'Switch View: Video ⇄ 3D Mesh ⇄ Split',
      icon: 'scan',
    },
    {
      id: 3,
      badge: 'STEP 03 / VOCALIZATION & TAPE',
      title: 'Live Transcript & Voice Synthesis',
      desc: 'Decoded words immediately append to the Live Transcript Tape. Click any word to inspect beam-search alternatives or correct with 1-click. Toggle Text-to-Speech (TTS) to vocalize your silent speech.',
      hotkey: 'T',
      hotkeyAction: 'Toggle Speech Audio Synthesis (TTS)',
      icon: 'volume',
    },
  ]

  const cur = steps[step - 1]

  return (
    <div className="ws-modal-backdrop" role="dialog" aria-modal="true" aria-label="Studio Quick-Start Guide">
      <div className="ws-quickstart-card" style={{
        background: '#0d1019',
        border: '1px solid #273147',
        borderRadius: '12px',
        maxWidth: '520px',
        width: '90%',
        padding: '24px',
        boxShadow: '0 16px 48px rgba(0, 0, 0, 0.7)',
        position: 'relative'
      }}>
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
          <div>
            <span style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', fontWeight: 700, color: '#38bdf8', letterSpacing: '0.08em' }}>
              {cur.badge}
            </span>
            <h3 style={{ margin: '4px 0 0', fontSize: '18px', color: '#f1f5f9', letterSpacing: '-0.02em' }}>
              {cur.title}
            </h3>
          </div>
          <button
            type="button"
            className="ws-btn-icon-sm"
            onClick={onClose}
            aria-label="Close guide"
            style={{ fontSize: '16px', color: '#94a3b8' }}
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.6, margin: '0 0 20px' }}>
          {cur.desc}
        </p>

        {/* Hotkey Tip Box */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          background: '#090c14',
          border: '1px solid #1a2233',
          borderRadius: '8px',
          padding: '10px 14px',
          marginBottom: '24px'
        }}>
          <kbd style={{
            background: '#1c2438',
            border: '1px solid #2d3748',
            borderRadius: '4px',
            padding: '2px 7px',
            fontSize: '11px',
            color: '#38bdf8',
            fontFamily: 'var(--font-mono)',
            fontWeight: 700
          }}>
            {cur.hotkey}
          </kbd>
          <span style={{ fontSize: '12px', color: '#cbd5e1' }}>
            {cur.hotkeyAction}
          </span>
        </div>

        {/* Step Indicators & Action Buttons */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            {steps.map(s => (
              <span
                key={s.id}
                style={{
                  width: s.id === step ? '20px' : '6px',
                  height: '6px',
                  borderRadius: '3px',
                  background: s.id === step ? '#38bdf8' : '#232d42',
                  transition: 'all 0.2s ease'
                }}
              />
            ))}
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            {step > 1 && (
              <button
                type="button"
                className="ws-btn ws-btn-ghost ws-btn-sm"
                onClick={handlePrev}
              >
                ◀ Back
              </button>
            )}
            <button
              type="button"
              className="ws-btn ws-btn-primary ws-btn-sm"
              onClick={handleNext}
            >
              {step === 3 ? 'Open Studio 🚀' : 'Next Step ▶'}
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
