import { useEffect } from 'react'
import { predictNextWords, type NextWordSuggestion } from '../../live-demo/languageModel'

interface PredictiveRibbonProps {
  transcriptWords: string[]
  onSelectWord: (word: string) => void
  disabled?: boolean
}

export function PredictiveRibbon({ transcriptWords, onSelectWord, disabled = false }: PredictiveRibbonProps) {
  const suggestions: NextWordSuggestion[] = predictNextWords(transcriptWords, 4)

  // Keyboard shortcut listener for keys 1 to 4
  useEffect(() => {
    if (disabled) return
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if user is typing in an input or textarea
      if (['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement)?.tagName)) return
      const keyIdx = parseInt(e.key, 10)
      if (keyIdx >= 1 && keyIdx <= 4 && suggestions[keyIdx - 1]) {
        e.preventDefault()
        onSelectWord(suggestions[keyIdx - 1].word)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [suggestions, onSelectWord, disabled])

  return (
    <div className="ws-predictive-ribbon" style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '8px 12px',
      background: '#0D1019',
      border: '1px solid #1F2536',
      borderRadius: '6px',
      marginBottom: '8px',
      fontSize: '11px',
      fontFamily: 'monospace',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        color: '#64748B',
        fontSize: '10px',
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
        paddingRight: '6px',
        borderRight: '1px solid #1F2536',
        flexShrink: 0,
      }}>
        <span style={{ color: '#38BDF8' }}>✦</span>
        <span>Next Word</span>
      </div>

      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        flexWrap: 'wrap',
        flex: 1,
      }}>
        {suggestions.map((item, idx) => (
          <button
            key={item.word}
            type="button"
            disabled={disabled}
            onClick={() => onSelectWord(item.word)}
            title={`Insert '${item.word}' (or press [${idx + 1}])`}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '6px',
              padding: '4px 10px',
              background: '#121622',
              border: '1px solid #1F2536',
              borderRadius: '4px',
              color: '#F1F5F9',
              cursor: disabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s cubic-bezier(0.16, 1, 0.3, 1)',
            }}
            onMouseEnter={e => {
              if (!disabled) {
                e.currentTarget.style.borderColor = '#38BDF8'
                e.currentTarget.style.background = '#161D2D'
              }
            }}
            onMouseLeave={e => {
              if (!disabled) {
                e.currentTarget.style.borderColor = '#1F2536'
                e.currentTarget.style.background = '#121622'
              }
            }}
          >
            <span style={{
              background: '#1F2536',
              color: '#38BDF8',
              padding: '1px 4px',
              borderRadius: '2px',
              fontSize: '9px',
              fontWeight: 700,
            }}>
              {idx + 1}
            </span>
            <span style={{ fontWeight: 600, letterSpacing: '0.04em' }}>{item.word}</span>
            <span style={{ color: '#64748B', fontSize: '10px', fontVariantNumeric: 'tabular-nums' }}>
              {Math.round(item.probability * 100)}%
            </span>
          </button>
        ))}
      </div>

      <div style={{
        color: '#475569',
        fontSize: '10px',
        flexShrink: 0,
      }}>
        Press [1]–[4]
      </div>
    </div>
  )
}
