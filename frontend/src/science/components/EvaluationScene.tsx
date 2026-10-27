import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'

export function EvaluationScene() {
  const [selectedCell, setSelectedCell] = useState<{ actual: string; predicted: string; count: number; type: 'TP' | 'FP' | 'FN' | 'TN' } | null>({
    actual: 'A',
    predicted: 'A',
    count: 142,
    type: 'TP'
  })

  const classes = ['A', 'B', 'C', 'D', 'E']

  // 5x5 sample confusion matrix (illustrative benchmark based on single-speaker baseline)
  const matrix: number[][] = [
    [142,  4,  2,  5,  3], // Actual A
    [  3, 128,  1,  6, 18], // Actual B (confuses with E/D)
    [  2,  2, 136, 12,  4], // Actual C (confuses with D)
    [  4,  5,  9, 131,  7], // Actual D
    [  3, 14,  3,  6, 130], // Actual E
  ]

  // Calculate binary stats for a chosen class (default A)
  const classIdx = 0 // Class A
  const TP = matrix[classIdx][classIdx]
  const FN = matrix[classIdx].reduce((acc, v, i) => i !== classIdx ? acc + v : acc, 0)
  const FP = matrix.reduce((acc, row, i) => i !== classIdx ? acc + row[classIdx] : acc, 0)
  const TN = matrix.reduce((acc, row, i) => {
    if (i === classIdx) return acc
    return acc + row.reduce((rAcc, v, j) => j !== classIdx ? rAcc + v : rAcc, 0)
  }, 0)

  const accuracy = (((TP + TN) / (TP + TN + FP + FN)) * 100).toFixed(1)
  const precision = ((TP / (TP + FP)) * 100).toFixed(1)
  const recall = ((TP / (TP + FN)) * 100).toFixed(1)
  const f1 = (2 * ((parseFloat(precision) * parseFloat(recall)) / (parseFloat(precision) + parseFloat(recall)))).toFixed(1)

  // Synthetic loss curve progression across 20 epochs
  const lossHistory = [
    { epoch: 1, trainLoss: 3.24, valLoss: 3.12 },
    { epoch: 4, trainLoss: 2.15, valLoss: 2.22 },
    { epoch: 8, trainLoss: 1.42, valLoss: 1.58 },
    { epoch: 12, trainLoss: 0.95, valLoss: 1.25 },
    { epoch: 16, trainLoss: 0.62, valLoss: 1.18 },
    { epoch: 20, trainLoss: 0.38, valLoss: 1.24 }, // slight overfitting inflection
  ]

  return (
    <section id="scene-evaluation" className="science-scene" aria-label="Scene 7: Measure the Model, Not the Marketing">
      <div className="scene-header">
        <span className="scene-num">Scene 07 · Empirical Rigor</span>
        <h2 className="scene-title">Measure the Model, Not the Marketing.</h2>
        <p className="scene-lead">
          AI demos often hide failure modes behind curated cherry-picked clips. Scientific computer vision demands 
          transparent confusion matrices, held-out speaker splits, and explicit accounting of class balance.
        </p>
      </div>

      <div className="sci-card">
        <div className="sci-card-header">
          <div className="sci-card-title">
            <span>5-Class Confusion Matrix & Per-Class Metrics</span>
          </div>
          <span className="sci-tag illustrative">Illustrative Fixture (GRID s1 Sample)</span>
        </div>

        <div className="sci-split">
          {/* Left: Interactive 5x5 Confusion Matrix */}
          <div>
            <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)', color: 'var(--ink-dim)', marginBottom: '0.8rem' }}>
              ACTUAL (ROW) VS PREDICTED (COLUMN) — HOVER CELLS:
            </div>

            <div style={{
              display: 'grid',
              gridTemplateColumns: `auto repeat(${classes.length}, 1fr)`,
              gap: '4px',
              fontFamily: 'var(--font-mono)',
              fontSize: '0.78rem'
            }}>
              {/* Header row */}
              <div style={{ padding: '6px' }} />
              {classes.map((c) => (
                <div key={`col-${c}`} style={{ textAlign: 'center', color: 'var(--accent)', padding: '4px' }}>
                  P({c})
                </div>
              ))}

              {/* Rows */}
              {classes.map((actualClass, r) => (
                <>
                  <div key={`row-${actualClass}`} style={{ color: 'var(--ink)', padding: '6px 8px', display: 'flex', alignItems: 'center' }}>
                    T({actualClass})
                  </div>
                  {classes.map((predClass, c) => {
                    const count = matrix[r][c]
                    const isDiagonal = r === c
                    const isHovered = selectedCell?.actual === actualClass && selectedCell?.predicted === predClass
                    const cellType: 'TP' | 'FP' | 'FN' | 'TN' = isDiagonal ? 'TP' : r === 0 ? 'FN' : c === 0 ? 'FP' : 'TN'

                    return (
                      <div
                        key={`cell-${r}-${c}`}
                        onMouseEnter={() => setSelectedCell({ actual: actualClass, predicted: predClass, count, type: cellType })}
                        style={{
                          aspectRatio: '1',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          background: isDiagonal
                            ? `rgba(143, 223, 203, ${0.15 + (count / 150) * 0.4})`
                            : count > 10
                            ? 'rgba(252, 211, 77, 0.15)'
                            : 'rgba(237, 235, 228, 0.02)',
                          border: isHovered
                            ? '1px solid var(--ink)'
                            : isDiagonal
                            ? '1px solid rgba(143, 223, 203, 0.3)'
                            : '1px solid var(--line)',
                          borderRadius: '4px',
                          color: isDiagonal ? 'var(--accent)' : count > 10 ? 'var(--warning)' : 'var(--ink-dim)',
                          cursor: 'pointer',
                          fontWeight: isDiagonal ? 600 : 400,
                          transform: isHovered ? 'scale(1.08)' : 'none',
                          transition: 'all 0.15s ease'
                        }}
                      >
                        {count}
                      </div>
                    )
                  })}
                </>
              ))}
            </div>

            <div style={{ fontSize: '0.72rem', color: 'var(--ink-faint)', marginTop: '0.8rem', fontFamily: 'var(--font-mono)' }}>
              Note: Diagonal green cells denote True Positives; off-diagonal yellow cells denote viseme misclassifications (e.g. B ↔ E).
            </div>
          </div>

          {/* Right: Formulas & Live Derived Metrics for Class 'A' */}
          <div>
            <div className="math-box">
              <div className="math-formula" style={{ fontSize: '0.95rem' }}>
                <MathExpression tex={EQUATIONS.f1}/>
              </div>
              <p className="math-desc">
                Harmonic mean balancing False Positives (over-triggering) against False Negatives (missed phonemes).
              </p>
            </div>

            {/* Calculated Statistics Grid */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '0.75rem',
              fontFamily: 'var(--font-mono)',
              marginTop: '1rem'
            }}>
              <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--line)' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>A VS REST ACCURACY</div>
                <div style={{ fontSize: '1.25rem', color: 'var(--accent)', fontWeight: 600 }}>{accuracy}%</div>
                <div className="math-metric-definition"><MathExpression tex={String.raw`\frac{\mathrm{TP}+\mathrm{TN}}{N}`} inline/></div>
              </div>

              <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--line)' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>CLASS 'A' PRECISION</div>
                <div style={{ fontSize: '1.25rem', color: 'var(--info)', fontWeight: 600 }}>{precision}%</div>
                <div className="math-metric-definition"><MathExpression tex={EQUATIONS.precision} inline/></div>
              </div>

              <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--line)' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>CLASS 'A' RECALL</div>
                <div style={{ fontSize: '1.25rem', color: 'var(--ink)', fontWeight: 600 }}>{recall}%</div>
                <div className="math-metric-definition"><MathExpression tex={EQUATIONS.recall} inline/></div>
              </div>

              <div style={{ background: 'rgba(237, 235, 228, 0.02)', padding: '0.8rem', borderRadius: '8px', border: '1px solid var(--line)' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>CLASS 'A' F1-SCORE</div>
                <div style={{ fontSize: '1.25rem', color: 'var(--success)', fontWeight: 600 }}>{f1}%</div>
                <div style={{ fontSize: '0.65rem', color: 'var(--ink-dim)' }}>Harmonic Balance</div>
              </div>
            </div>

            {/* Selected Cell Inspector */}
            {selectedCell && (
              <div style={{
                marginTop: '1rem',
                padding: '0.8rem',
                borderRadius: '6px',
                background: 'rgba(143, 223, 203, 0.05)',
                border: '1px solid rgba(143, 223, 203, 0.2)',
                fontSize: '0.78rem',
                fontFamily: 'var(--font-mono)'
              }}>
                Target <strong>'{selectedCell.actual}'</strong> classified as <strong>'{selectedCell.predicted}'</strong>: 
                <span style={{ color: 'var(--accent)', marginLeft: '6px' }}>{selectedCell.count} instances ({selectedCell.type})</span>
              </div>
            )}
          </div>
        </div>

        {/* Loss Convergence & Generalization Sub-panel */}
        <div style={{ borderTop: '1px solid var(--line)', marginTop: '2rem', paddingTop: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.82rem', fontFamily: 'var(--font-mono)', color: 'var(--ink)', fontWeight: 600 }}>
              Training Loss vs Validation Loss (CosineAnnealingLR Schedule)
            </span>
            <div style={{ display: 'flex', gap: '1rem', fontSize: '0.72rem', fontFamily: 'var(--font-mono)' }}>
              <span style={{ color: 'var(--accent)' }}>● Train Loss</span>
              <span style={{ color: 'var(--info)' }}>● Validation Loss</span>
            </div>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(6, 1fr)',
            gap: '0.5rem',
            background: '#070709',
            padding: '1rem',
            borderRadius: '8px',
            border: '1px solid var(--line)'
          }}>
            {lossHistory.map((pt) => (
              <div key={pt.epoch} style={{ textAlign: 'center', fontFamily: 'var(--font-mono)' }}>
                <div style={{ fontSize: '0.68rem', color: 'var(--ink-faint)' }}>Ep {pt.epoch}</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--accent)', margin: '2px 0' }}>{pt.trainLoss}</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--info)' }}>{pt.valLoss}</div>
              </div>
            ))}
          </div>
          <div style={{ fontSize: '0.74rem', color: 'var(--ink-dim)', marginTop: '0.6rem' }}>
            Inflection Analysis: Early stopping triggered when validation loss ceases to decrease for 3 consecutive epochs, 
            preventing memorization of specific speaker idiosyncratic facial wrinkles.
          </div>
        </div>
      </div>
    </section>
  )
}
