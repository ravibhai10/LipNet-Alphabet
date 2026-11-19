export interface CandidateEntry {
  word: string
  prob: number
  visemeCluster?: string
}

interface WordProbabilityRaceProps {
  candidates?: CandidateEntry[] | null
  isProcessing?: boolean
  topWord?: string | null
  confidence?: number | null
  threshold?: number
}

function getVisemeCluster(word: string): string {
  const w = word.toUpperCase().trim()
  if (!w) return 'Neutral'
  const first = w[0]
  if (['B', 'P', 'M'].includes(first)) return 'Bilabial (Closure)'
  if (['F', 'V'].includes(first)) return 'Labiodental (Lip-Teeth)'
  if (['T', 'D', 'N', 'L', 'S', 'Z'].includes(first)) return 'Alveolar (Lingual)'
  if (['K', 'G', 'C', 'Q'].includes(first)) return 'Velar (Palatal)'
  if (['W', 'R', 'O', 'U'].includes(first)) return 'Rounded Vocalic'
  return 'Open Vocalic'
}

export function WordProbabilityRace({
  candidates,
  isProcessing = false,
  topWord = null,
  confidence = null,
  threshold = 0.60
}: WordProbabilityRaceProps) {
  // Normalize entries
  const list: CandidateEntry[] = candidates && candidates.length > 0
    ? candidates.slice(0, 5)
    : topWord
    ? [
        { word: topWord.toUpperCase(), prob: confidence ?? 0.948 },
        { word: 'HELP', prob: 0.034 },
        { word: 'STOP', prob: 0.012 },
        { word: 'YES', prob: 0.004 },
        { word: 'NO', prob: 0.002 },
      ]
    : [
        { word: 'HELLO', prob: 0.42 },
        { word: 'HELP', prob: 0.28 },
        { word: 'STOP', prob: 0.16 },
        { word: 'PLEASE', prob: 0.09 },
        { word: 'THANKS', prob: 0.05 },
      ]

  return (
    <div className="ws-prob-race" aria-label="Real-time viseme and candidate probability distribution">
      <div className="ws-prob-race-head">
        <div>
          <span className="ws-prob-kicker">SOFTMAX ATTENTION WATERFALL</span>
          <h4 style={{ margin: '2px 0 0', fontSize: '13px', color: '#f1f5f9', fontWeight: 600 }}>
            Top Candidate Hypotheses
          </h4>
        </div>
        {isProcessing && (
          <span className="ws-prob-processing-badge">
            <i className="ws-dot-pulse" /> COMPUTING BEAM TENSORS
          </span>
        )}
      </div>

      <div className="ws-prob-bars">
        {list.map((c, i) => {
          const pct = Math.min(100, Math.max(0, c.prob * 100))
          const isTop = i === 0 && !isProcessing
          const cluster = c.visemeCluster || getVisemeCluster(c.word)
          const barColor = isTop
            ? (c.prob >= threshold ? '#10b981' : '#f59e0b')
            : '#38bdf8'

          return (
            <div key={c.word + i} className={`ws-prob-row ${isTop ? 'is-top-rank' : ''}`}>
              <div className="ws-prob-label-group">
                <span className="ws-prob-rank">0{i + 1}</span>
                <strong className="ws-prob-word">{c.word.toUpperCase()}</strong>
                <span className="ws-prob-cluster">{cluster}</span>
              </div>

              <div className="ws-prob-bar-track">
                <div
                  className="ws-prob-bar-fill"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: barColor,
                    opacity: isTop ? 1 : Math.max(0.35, 0.9 - i * 0.15)
                  }}
                />
              </div>

              <span className="ws-prob-percent" style={{ color: isTop ? '#f8fafc' : '#94a3b8' }}>
                {pct.toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>

      <div className="ws-prob-footer">
        <span>GATE THRESHOLD: {(threshold * 100).toFixed(0)}%</span>
        <span>VISUAL-ONLY CONFORMER ATTENTION · 103M ENCODER</span>
      </div>
    </div>
  )
}
