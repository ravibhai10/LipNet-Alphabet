import { useState, useEffect, useRef, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { cyberAudio } from '../../utils/CyberSoundFX'
import { Icon } from '../Icon'

interface CommandItem {
  id: string
  title: string
  subtitle: string
  category: 'NAVIGATION' | 'SYSTEM CONTROLS' | 'AI ACTIONS'
  icon: string
  shortcut?: string
  action: () => void
}

interface Props {
  isOpen: boolean
  onClose: () => void
  onToggleTheme?: () => void
  onRefreshBackend?: () => void
}

export function CyberCommandPalette({ isOpen, onClose, onToggleTheme, onRefreshBackend }: Props) {
  const [query, setQuery] = useState('')
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [audioEnabled, setAudioEnabled] = useState(() => cyberAudio.isEnabled())
  const navigate = useNavigate()
  const inputRef = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)

  const commands: CommandItem[] = useMemo(() => [
    {
      id: 'nav-overview',
      title: 'Overview & Mission Control',
      subtitle: 'System readiness, benchmark claims, and live telemetry',
      category: 'NAVIGATION',
      icon: 'grid',
      shortcut: 'G O',
      action: () => { navigate('/demo/overview'); onClose() }
    },
    {
      id: 'nav-live',
      title: 'Live Lip Reading Studio',
      subtitle: 'Real-time camera visual speech recognition & MAR tracking',
      category: 'NAVIGATION',
      icon: 'camera',
      shortcut: 'G L',
      action: () => { navigate('/demo/live'); onClose() }
    },
    {
      id: 'nav-feedback',
      title: 'Adaptive Feedback Lab',
      subtitle: 'Per-user adaptation registry, calibration, and test suites',
      category: 'NAVIGATION',
      icon: 'activity',
      action: () => { navigate('/demo/feedback'); onClose() }
    },
    {
      id: 'nav-collect',
      title: 'Word Collection & Dataset Expansion',
      subtitle: 'MIRACL-8 isolated word recording with automated QA checks',
      category: 'NAVIGATION',
      icon: 'camera',
      action: () => { navigate('/demo/collect'); onClose() }
    },
    {
      id: 'nav-pipeline',
      title: 'Neural Pipeline & Architecture',
      subtitle: 'Interactive 3D-CNN + 12-layer AV-HuBERT Transformer tensor flow',
      category: 'NAVIGATION',
      icon: 'pipeline',
      action: () => { navigate('/demo/pipeline'); onClose() }
    },
    {
      id: 'nav-testing',
      title: 'Evaluation & Verification Suites',
      subtitle: 'Run and inspect 73 vitest suites, speaker-disjoint F1 metrics',
      category: 'NAVIGATION',
      icon: 'chart',
      shortcut: 'G T',
      action: () => { navigate('/demo/testing'); onClose() }
    },
    {
      id: 'nav-models',
      title: 'Model Checkpoint Registry',
      subtitle: 'AV-HuBERT L9 (94.79% Top-1) & CNN-BiLSTM experimental weights',
      category: 'NAVIGATION',
      icon: 'layers',
      shortcut: 'G M',
      action: () => { navigate('/demo/models'); onClose() }
    },
    {
      id: 'nav-dataset',
      title: 'Dataset Explorer',
      subtitle: '32,891 MIRACL lip recordings across 33 speakers',
      category: 'NAVIGATION',
      icon: 'database',
      shortcut: 'G D',
      action: () => { navigate('/demo/dataset'); onClose() }
    },
    {
      id: 'nav-docs',
      title: 'Documentation & Audit Resolutions',
      subtitle: 'Complete D1-D5 resolution policies, math specs, and API contracts',
      category: 'NAVIGATION',
      icon: 'book',
      action: () => { navigate('/demo/docs'); onClose() }
    },
    {
      id: 'action-audio-toggle',
      title: audioEnabled ? 'Mute Audio FX' : 'Enable Audio FX',
      subtitle: audioEnabled ? 'Mute procedural interface feedback sounds' : 'Enable subtle acoustic interaction feedback',
      category: 'SYSTEM CONTROLS',
      icon: audioEnabled ? 'volume' : 'volume-x',
      shortcut: 'M',
      action: () => {
        const next = cyberAudio.toggle()
        setAudioEnabled(next)
      }
    },
    {
      id: 'action-theme-toggle',
      title: 'Toggle Theme (Dark / Light)',
      subtitle: 'Switch between dark and light workspace appearance',
      category: 'SYSTEM CONTROLS',
      icon: 'sun',
      shortcut: 'T',
      action: () => {
        if (onToggleTheme) onToggleTheme()
        onClose()
      }
    },
    {
      id: 'action-lrw-explorer',
      title: 'Open LRW-500 Corpus & Viseme Explorer',
      subtitle: '500 English wild words grouped by phonetic viseme onset clusters',
      category: 'AI ACTIONS',
      icon: 'database',
      action: () => { navigate('/demo/dataset'); onClose() }
    },
    {
      id: 'action-tts-synthesis',
      title: 'Vocalize Silent Speech (TTS Engine)',
      subtitle: 'Trigger Web Speech API audio synthesis for live decoded words',
      category: 'AI ACTIONS',
      icon: 'volume',
      shortcut: 'T',
      action: () => {
        if ('speechSynthesis' in window) {
          const u = new SpeechSynthesisUtterance('LipNet AI voice synthesis active')
          window.speechSynthesis.speak(u)
        }
        onClose()
      }
    },
    {
      id: 'action-refresh-backend',
      title: 'Ping / Reconnect Backend Daemon',
      subtitle: 'Check http://127.0.0.1:8001/api/health status',
      category: 'AI ACTIONS',
      icon: 'refresh',
      shortcut: 'R',
      action: () => {
        if (onRefreshBackend) onRefreshBackend()
        cyberAudio.playClick()
        onClose()
      }
    },
    {
      id: 'action-test-sound',
      title: 'Test Audio Feedback',
      subtitle: 'Trigger acoustic feedback cues for lock and recognition',
      category: 'AI ACTIONS',
      icon: 'activity',
      action: () => {
        cyberAudio.playTargetLock()
        setTimeout(() => cyberAudio.playDecodeWord(), 150)
      }
    }
  ], [navigate, onClose, onToggleTheme, onRefreshBackend, audioEnabled])

  const filtered = useMemo(() => {
    if (!query.trim()) return commands
    const q = query.toLowerCase().trim()
    return commands.filter(c => 
      c.title.toLowerCase().includes(q) || 
      c.subtitle.toLowerCase().includes(q) || 
      c.category.toLowerCase().includes(q)
    )
  }, [commands, query])

  useEffect(() => {
    setSelectedIndex(0)
  }, [query])

  useEffect(() => {
    if (isOpen) {
      cyberAudio.playPaletteOpen()
      setQuery('')
      setSelectedIndex(0)
      setTimeout(() => inputRef.current?.focus(), 50)
    }
  }, [isOpen])

  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault()
        onClose()
      } else if (e.key === 'ArrowDown') {
        e.preventDefault()
        setSelectedIndex(prev => (prev + 1) % (filtered.length || 1))
      } else if (e.key === 'ArrowUp') {
        e.preventDefault()
        setSelectedIndex(prev => (prev - 1 + (filtered.length || 1)) % (filtered.length || 1))
      } else if (e.key === 'Enter') {
        e.preventDefault()
        if (filtered[selectedIndex]) {
          cyberAudio.playClick()
          filtered[selectedIndex].action()
        }
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, filtered, selectedIndex, onClose])

  if (!isOpen) return null

  return (
    <div 
      className="cyber-palette-backdrop"
      onClick={onClose}
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        backgroundColor: 'rgba(0, 0, 0, 0.6)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        display: 'flex',
        alignItems: 'flex-start',
        justifyContent: 'center',
        paddingTop: '14vh',
        animation: 'fadeIn 0.15s ease'
      }}
    >
      <div 
        className="cyber-palette-window"
        onClick={e => e.stopPropagation()}
        style={{
          width: '92%',
          maxWidth: '640px',
          background: 'rgba(14, 18, 28, 0.94)',
          backdropFilter: 'blur(28px)',
          WebkitBackdropFilter: 'blur(28px)',
          border: '1px solid rgba(99, 102, 241, 0.35)',
          boxShadow: '0 24px 70px rgba(0, 0, 0, 0.8), 0 0 35px rgba(99, 102, 241, 0.2)',
          borderRadius: '14px',
          overflow: 'hidden',
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '72vh'
        }}
      >
        {/* Search Header */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          padding: '14px 18px',
          borderBottom: '1px solid var(--ws-border)',
          background: 'var(--ws-surface)'
        }}>
          <span style={{ color: '#818cf8', display: 'flex', filter: 'drop-shadow(0 0 6px #6366f1)' }}>
            <Icon name="search" size={17} />
          </span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Type a command or jump to workspace module... (ESC to close)"
            style={{
              flex: 1,
              background: 'transparent',
              border: 'none',
              outline: 'none',
              fontFamily: 'inherit',
              fontSize: '14px',
              color: 'var(--ws-text)',
              fontWeight: 450
            }}
          />
          <kbd style={{
            fontFamily: 'var(--font-mono)',
            fontSize: '10px',
            color: 'var(--ws-text-dim)',
            background: 'var(--ws-surface2)',
            padding: '2px 6px',
            borderRadius: '4px',
            border: '1px solid var(--ws-border)'
          }}>ESC</kbd>
        </div>

        {/* Command List */}
        <div 
          ref={listRef}
          style={{
            overflowY: 'auto',
            padding: '8px',
            flex: 1
          }}
        >
          {filtered.length === 0 ? (
            <div style={{ padding: '36px 20px', textAlign: 'center', color: 'var(--ws-text-muted)' }}>
              <p style={{ margin: 0, fontSize: '13px' }}>No commands found matching "{query}"</p>
            </div>
          ) : (
            filtered.map((cmd, idx) => {
              const isSelected = idx === selectedIndex
              return (
                <div
                  key={cmd.id}
                  onClick={() => {
                    cyberAudio.playClick()
                    cmd.action()
                  }}
                  onMouseEnter={() => {
                    setSelectedIndex(idx)
                  }}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '8px 12px',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    background: isSelected ? 'linear-gradient(90deg, rgba(99, 102, 241, 0.18), rgba(56, 189, 248, 0.08))' : 'transparent',
                    border: isSelected ? '1px solid rgba(99, 102, 241, 0.4)' : '1px solid transparent',
                    transition: 'background 0.15s ease',
                    marginBottom: '2px'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '11px' }}>
                    <div style={{
                      width: '28px',
                      height: '28px',
                      borderRadius: '6px',
                      background: isSelected ? 'var(--ws-surface3)' : 'var(--ws-surface2)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isSelected ? 'var(--ws-accent)' : 'var(--ws-text-muted)'
                    }}>
                      <Icon name={cmd.icon} size={15} />
                    </div>
                    <div>
                      <div style={{
                        fontSize: '13px',
                        fontWeight: 500,
                        color: 'var(--ws-text)',
                        letterSpacing: '-0.01em'
                      }}>
                        {cmd.title}
                      </div>
                      <div style={{
                        fontSize: '11px',
                        color: 'var(--ws-text-muted)',
                        marginTop: '1px'
                      }}>
                        {cmd.subtitle}
                      </div>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {cmd.shortcut && (
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '10px',
                        color: 'var(--ws-text-dim)',
                        background: 'var(--ws-surface2)',
                        border: '1px solid var(--ws-border)',
                        borderRadius: '4px',
                        padding: '2px 5px'
                      }}>
                        {cmd.shortcut}
                      </span>
                    )}
                    {isSelected && (
                      <span style={{
                        fontFamily: 'var(--font-mono)',
                        fontSize: '11px',
                        color: 'var(--ws-accent)',
                        marginLeft: '4px'
                      }}>
                        ↵
                      </span>
                    )}
                  </div>
                </div>
              )
            })
          )}
        </div>

        {/* Footer Telemetry */}
        <div style={{
          padding: '8px 16px',
          borderTop: '1px solid var(--ws-border)',
          background: 'var(--ws-surface2)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          fontSize: '11px',
          fontFamily: 'var(--font-mono)',
          color: 'var(--ws-text-dim)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
            <span><strong style={{ color: 'var(--ws-text)' }}>↑↓</strong> Navigate</span>
            <span><strong style={{ color: 'var(--ws-text)' }}>↵</strong> Select</span>
            <span><strong style={{ color: 'var(--ws-text)' }}>ESC</strong> Close</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--ws-success)' }} />
            <span>AV-HuBERT L9 · MIRACL-8</span>
          </div>
        </div>
      </div>
    </div>
  )
}
