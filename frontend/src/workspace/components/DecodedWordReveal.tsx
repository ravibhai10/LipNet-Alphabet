import { useEffect, useState } from 'react'

interface DecodedWordRevealProps {
  targetWord: string | null
  isProcessing?: boolean
  isUncertain?: boolean
}

const GLYPHS = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789'

export function DecodedWordReveal({
  targetWord,
  isProcessing = false,
  isUncertain = false,
}: DecodedWordRevealProps) {
  const [displayText, setDisplayText] = useState<string>(targetWord ?? '—')

  useEffect(() => {
    if (!targetWord || targetWord === '—') {
      setDisplayText('—')
      return
    }

    let iteration = 0
    const target = targetWord.toUpperCase()
    const maxIterations = target.length * 2 + 2

    const interval = window.setInterval(() => {
      setDisplayText(
        target
          .split('')
          .map((_, index) => {
            if (index < iteration / 2) {
              return target[index]
            }
            return GLYPHS[Math.floor(Math.random() * GLYPHS.length)]
          })
          .join('')
      )

      if (iteration >= maxIterations) {
        setDisplayText(target)
        clearInterval(interval)
      }

      iteration += 1
    }, 30)

    return () => clearInterval(interval)
  }, [targetWord])

  if (isProcessing) {
    return (
      <span style={{ color: 'var(--ws-text-muted)', fontFamily: 'var(--font-mono)', fontSize: '24px' }}>
        ANALYZING…
      </span>
    )
  }

  if (isUncertain) {
    return (
      <span style={{ color: 'var(--ws-warning)', fontFamily: 'var(--font-mono)', fontSize: '24px', fontWeight: 600 }}>
        UNCERTAIN
      </span>
    )
  }

  return (
    <span
      style={{
        fontFamily: 'var(--font-mono)',
        fontWeight: 650,
        letterSpacing: '0.04em',
        color: 'var(--ws-text)',
        fontSize: '28px'
      }}
    >
      {displayText}
    </span>
  )
}
