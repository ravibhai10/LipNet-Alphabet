import { useState, useEffect, useRef, useCallback } from 'react'
import { Icon } from '../Icon'
import { cyberAudio } from '../../utils/CyberSoundFX'
import { PredictiveRibbon } from './PredictiveRibbon'

export interface TranscriptWord {
  id: string
  word: string
  confidence: number
  alternatives: Array<{ label: string; score: number }>
  timestamp: number
}

interface TranscriptComposerProps {
  latestWord: string | null
  confidence: number | null
  alternatives?: Array<{ label: string; score: number }> | null
  ttsEnabled?: boolean
  onToggleTTS?: () => void
  onWordCorrected?: (original: string, corrected: string) => void
}

export function TranscriptComposer({
  latestWord,
  confidence,
  alternatives = [],
  ttsEnabled = false,
  onToggleTTS,
  onWordCorrected,
}: TranscriptComposerProps) {
  const [tape, setTape] = useState<TranscriptWord[]>([
    // Initial welcoming placeholder sequence
    { id: 'init-1', word: 'HELLO', confidence: 0.948, alternatives: [{ label: 'HELLO', score: 0.948 }, { label: 'HELP', score: 0.032 }], timestamp: Date.now() - 10000 },
    { id: 'init-2', word: 'LIPNET', confidence: 0.912, alternatives: [{ label: 'LIPNET', score: 0.912 }, { label: 'LINEAR', score: 0.045 }], timestamp: Date.now() - 5000 },
  ])
  const [activeWordId, setActiveWordId] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)
  const [manualCorrection, setManualCorrection] = useState('')
  const lastProcessedWordRef = useRef<string | null>(null)

  // Web Speech API Voice Synthesis
  const speakText = useCallback((text: string) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return
    window.speechSynthesis.cancel()
    const utterance = new SpeechSynthesisUtterance(text.toLowerCase())
    utterance.rate = 0.95
    utterance.pitch = 1.0
    window.speechSynthesis.speak(utterance)
  }, [])

  // Ingest new word
  useEffect(() => {
    if (!latestWord || latestWord === '—' || latestWord === lastProcessedWordRef.current) return
    lastProcessedWordRef.current = latestWord

    const conf = confidence ?? 0.88
    const alts = alternatives && alternatives.length > 0 ? alternatives : [
      { label: latestWord, score: conf },
      { label: 'HELP', score: 0.04 },
      { label: 'STOP', score: 0.02 },
      { label: 'YES', score: 0.01 },
      { label: 'NO', score: 0.01 }
    ]

    const newEntry: TranscriptWord = {
      id: `w-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
      word: latestWord.toUpperCase(),
      confidence: conf,
      alternatives: alts,
      timestamp: Date.now()
    }

    setTape(prev => [...prev, newEntry])

    if (ttsEnabled) {
      speakText(latestWord)
    }
  }, [latestWord, confidence, alternatives, ttsEnabled, speakText])

  const handleCopy = () => {
    cyberAudio.playClick()
    const fullText = tape.map(w => w.word).join(' ')
    if (navigator.clipboard) {
      void navigator.clipboard.writeText(fullText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  const handleSpeakAll = () => {
    cyberAudio.playClick()
    const fullText = tape.map(w => w.word).join(' ')
    if (fullText.trim()) {
      speakText(fullText)
    }
  }

  const handleClear = () => {
    cyberAudio.playClick()
    setTape([])
    setActiveWordId(null)
  }

  const handleApplyAlternative = (targetId: string, newLabel: string) => {
    cyberAudio.playClick()
    const target = tape.find(w => w.id === targetId)
    if (target && onWordCorrected) {
      onWordCorrected(target.word, newLabel.toUpperCase())
    }
    setTape(prev => prev.map(w => w.id === targetId ? { ...w, word: newLabel.toUpperCase(), confidence: 1.0 } : w))
    setActiveWordId(null)
    setManualCorrection('')
  }

  const handleDeleteWord = (targetId: string) => {
    cyberAudio.playClick()
    setTape(prev => prev.filter(w => w.id !== targetId))
    setActiveWordId(null)
  }

  const activeWord = tape.find(w => w.id === activeWordId)
  const fullSentence = tape.map(w => w.word).join(' ')
  const avgConfidence = tape.length ? (tape.reduce((acc, w) => acc + w.confidence, 0) / tape.length) * 100 : 0

  return (
    <section className="ws-transcript-composer" aria-label="Live continuous transcript composer">
      {/* Top Header Deck */}
      <div className="ws-transcript-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="ws-transcript-badge-icon">
            <Icon name="activity" size={14} />
          </span>
          <h3 style={{ margin: 0, fontSize: '13px', fontWeight: 600, color: '#f1f5f9', letterSpacing: '-0.01em' }}>
            Live Speech Transcript Tape
          </h3>
          <span className="ws-transcript-counter">
            {tape.length} {tape.length === 1 ? 'WORD' : 'WORDS'}
          </span>
          {tape.length > 0 && (
            <span className="ws-transcript-confidence-chip">
              AVG CONF: {avgConfidence.toFixed(1)}%
            </span>
          )}
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          {onToggleTTS && (
            <button
              type="button"
              className={`ws-btn-tactile ws-transcript-action-btn ${ttsEnabled ? 'is-active-tts' : ''}`}
              onClick={() => {
                cyberAudio.playClick()
                onToggleTTS()
              }}
              title="Toggle Live Web Speech Audio Synthesis (Shortcut: T)"
            >
              <Icon name="volume" size={13} />
              <span>TTS: {ttsEnabled ? 'ON' : 'OFF'}</span>
              <kbd className="ws-kbd-micro">T</kbd>
            </button>
          )}

          <button
            type="button"
            className="ws-btn-tactile ws-transcript-action-btn"
            onClick={handleSpeakAll}
            disabled={tape.length === 0}
            title="Speak entire sentence aloud"
          >
            <Icon name="volume" size={13} />
            <span>Speak All</span>
          </button>

          <button
            type="button"
            className="ws-btn-tactile ws-transcript-action-btn"
            onClick={handleCopy}
            disabled={tape.length === 0}
            title="Copy accumulated transcript to clipboard"
          >
            <Icon name="layers" size={13} />
            <span>{copied ? 'Copied!' : 'Copy'}</span>
          </button>

          <button
            type="button"
            className="ws-btn-tactile ws-transcript-action-btn ws-btn-danger"
            onClick={handleClear}
            disabled={tape.length === 0}
            title="Clear transcript tape"
          >
            <Icon name="refresh" size={13} />
            <span>Clear</span>
          </button>
        </div>
      </div>

      {/* Predictive Language Model Ribbon */}
      <PredictiveRibbon
        transcriptWords={tape.map(w => w.word)}
        onSelectWord={(word) => {
          cyberAudio.playDecodeWord()
          const newEntry: TranscriptWord = {
            id: `pred-${Date.now()}-${Math.random().toString(36).substr(2, 4)}`,
            word: word.toUpperCase(),
            confidence: 0.95,
            alternatives: [{ label: word.toUpperCase(), score: 0.95 }],
            timestamp: Date.now(),
          }
          setTape(prev => [...prev, newEntry])
          if (ttsEnabled) {
            speakText(word)
          }
        }}
      />

      {/* Main Tape Viewport */}
      <div className="ws-transcript-viewport">
        {tape.length === 0 ? (
          <div className="ws-transcript-empty">
            <span style={{ opacity: 0.5 }}>Tape empty. Speak words or letters in the studio to compose sentences.</span>
          </div>
        ) : (
          <div className="ws-transcript-chips">
            {tape.map((item, index) => {
              const isEmerald = item.confidence >= 0.85
              const isAmber = item.confidence >= 0.60 && item.confidence < 0.85
              const colorClass = isEmerald ? 'chip-emerald' : isAmber ? 'chip-amber' : 'chip-rose'
              const isSelected = activeWordId === item.id

              return (
                <div key={item.id} style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                  <button
                    type="button"
                    className={`ws-transcript-chip ws-btn-tactile ${colorClass} ${isSelected ? 'is-selected' : ''}`}
                    onClick={() => {
                      cyberAudio.playClick()
                      setActiveWordId(isSelected ? null : item.id)
                    }}
                    title={`Click to inspect alternatives / correct (${(item.confidence * 100).toFixed(1)}% confidence)`}
                  >
                    <span className="ws-chip-text">{item.word}</span>
                    <span className="ws-chip-conf">{(item.confidence * 100).toFixed(0)}%</span>
                  </button>
                  {index < tape.length - 1 && <span className="ws-chip-spacer">·</span>}
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Word Correction Popover */}
      {activeWord && (
        <div className="ws-transcript-popover" role="dialog" aria-label="Alternative word hypotheses">
          <div className="ws-popover-header">
            <div>
              <span className="ws-popover-kicker">BEAM SEARCH HYPOTHESES</span>
              <h4 style={{ margin: '2px 0 0', color: '#fff', fontSize: '13px' }}>
                Alternative Candidates for "{activeWord.word}"
              </h4>
            </div>
            <button
              type="button"
              className="ws-btn-icon-sm"
              onClick={() => setActiveWordId(null)}
              aria-label="Close popover"
            >
              ✕
            </button>
          </div>

          <div className="ws-popover-alts">
            {activeWord.alternatives.map(alt => (
              <button
                key={alt.label}
                type="button"
                className={`ws-alt-btn ws-btn-tactile ${alt.label.toUpperCase() === activeWord.word ? 'is-active-alt' : ''}`}
                onClick={() => handleApplyAlternative(activeWord.id, alt.label)}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span className="ws-alt-label">{alt.label.toUpperCase()}</span>
                  {alt.label.toUpperCase() === activeWord.word && (
                    <span className="ws-alt-badge">SELECTED</span>
                  )}
                </div>
                <span className="ws-alt-score">{(alt.score * 100).toFixed(1)}%</span>
              </button>
            ))}
          </div>

          {/* Manual Correction Input */}
          <div className="ws-popover-manual">
            <input
              type="text"
              placeholder="Or type manual correction..."
              value={manualCorrection}
              onChange={e => setManualCorrection(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && manualCorrection.trim()) {
                  handleApplyAlternative(activeWord.id, manualCorrection.trim())
                }
              }}
              className="ws-popover-input"
            />
            <button
              type="button"
              className="ws-btn ws-btn-primary ws-btn-sm"
              disabled={!manualCorrection.trim()}
              onClick={() => handleApplyAlternative(activeWord.id, manualCorrection.trim())}
            >
              Apply
            </button>
            <button
              type="button"
              className="ws-btn ws-btn-ghost ws-btn-sm ws-btn-danger"
              onClick={() => handleDeleteWord(activeWord.id)}
              title="Delete word from tape"
            >
              Delete
            </button>
          </div>
        </div>
      )}

      {/* Rendered Live Sentence Summary Bar */}
      {tape.length > 0 && (
        <div className="ws-transcript-footer">
          <span style={{ color: '#64748b', fontSize: '11px', fontFamily: 'var(--font-mono)' }}>COMPOSED SENTENCE:</span>
          <p className="ws-transcript-sentence">"{fullSentence}"</p>
        </div>
      )}
    </section>
  )
}
