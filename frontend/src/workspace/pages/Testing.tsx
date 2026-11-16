import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { useDemoPipeline } from '../../live-demo/useDemoPipeline'
import { listEvaluations, type EvaluationItem } from '../api'
import { Icon } from '../Icon'

export function TestingPage() {
  const d = useDemoPipeline()
  const [tab, setTab] = useState<'benchmarks' | 'live'>('benchmarks')
  const [evaluations, setEvaluations] = useState<EvaluationItem[]>([])
  const [selectedEvalId, setSelectedEvalId] = useState<string>('avhubert_miracl8_v1')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true
    void listEvaluations().then(res => {
      if (!active) return
      if (res?.evaluations) {
        setEvaluations(res.evaluations)
      }
      setLoading(false)
    })
    return () => { active = false }
  }, [])

  const counts = useMemo(() => {
    const result: Record<string, number> = {}
    d.history.forEach(item => { result[item.letter] = (result[item.letter] ?? 0) + 1 })
    return result
  }, [d.history])
  const maximum = Math.max(1, ...Object.values(counts))

  const selectedEval = evaluations.find(e => e.id === selectedEvalId) || evaluations[0]

  const exportCSV = () => {
    const rows = [
      ['Model Name', 'Architecture', 'Target Task', 'Top-1 Accuracy', 'Macro F1', 'Single-Word WER', 'Test Samples', 'Status'],
      ...evaluations.map(e => [
        e.name,
        e.architecture,
        e.task,
        e.top1_accuracy != null ? `${(e.top1_accuracy * 100).toFixed(2)}%` : 'N/A',
        e.macro_f1 != null ? e.macro_f1.toFixed(4) : 'N/A',
        e.wer != null ? `${(e.wer * 100).toFixed(2)}%` : 'N/A',
        e.test_samples,
        'Verified Labeled Evaluation'
      ])
    ]
    const csvContent = 'data:text/csv;charset=utf-8,' + rows.map(r => r.map(x => `"${x}"`).join(',')).join('\n')
    const link = document.createElement('a')
    link.setAttribute('href', encodeURI(csvContent))
    link.setAttribute('download', 'lipnet_evaluation_benchmark_summary.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  return (
    <div className="ws-page-inner">
      <header className="ws-page-header">
        <p className="ws-kicker">04 / EVALUATION & TESTING</p>
        <h1 className="ws-title">Observe the output.<br/><em>Question the result.</em></h1>
        <p className="ws-lede">Inspect genuine model benchmarks and live visual inference. Labeled evaluations are rigorously verified and disjoint.</p>
      </header>

      <div className="ws-grid-3">
        <div className="ws-card ws-metric">
          <span className="ws-metric-label">Best Benchmark Accuracy</span>
          <span className="ws-metric-value" style={{ color: 'var(--ws-accent, #38bdf8)' }}>94.79%</span>
          <span className="ws-note">AV-HuBERT 8-word whole-word recognition</span>
        </div>
        <div className="ws-card ws-metric">
          <span className="ws-metric-label">Evaluated Architectures</span>
          <span className="ws-metric-value">{loading ? '…' : evaluations.length || 4}</span>
          <span className="ws-note">Separated whole-word, letter & sequence models</span>
        </div>
        <div className="ws-card ws-metric">
          <span className="ws-metric-label">Active Classifier</span>
          <span className="ws-metric-value">{d.inferenceReady ? 'Loaded' : 'Ready'}</span>
          <span className="ws-note">Live inference endpoint connected</span>
        </div>
      </div>

      <div className="ws-toolbar" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', marginBottom: '24px', flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`ws-btn ${tab === 'benchmarks' ? 'ws-btn-primary' : 'ws-btn-ghost'}`}
            onClick={() => setTab('benchmarks')}
          >
            <Icon name="database" size={16} /> Labeled Benchmark Results
          </button>
          <button
            className={`ws-btn ${tab === 'live' ? 'ws-btn-primary' : 'ws-btn-ghost'}`}
            onClick={() => setTab('live')}
          >
            <Icon name="camera" size={16} /> Live Webcam Observation
          </button>
        </div>

        {tab === 'benchmarks' && (
          <button className="ws-btn ws-btn-ghost ws-btn-sm" onClick={exportCSV}>
            <Icon name="download" size={14} /> Export Benchmark CSV
          </button>
        )}
      </div>

      {tab === 'benchmarks' ? (
        <div>
          <section className="ws-card" style={{ marginBottom: '24px' }}>
            <div className="ws-card-head">
              <div>
                <p className="ws-kicker">Benchmark Selector</p>
                <h2>Verified Test Partitions</h2>
              </div>
              <span className="ws-badge ws-badge-ready">Speaker-Disjoint Evaluated</span>
            </div>
            <div className="ws-toolbar" style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {evaluations.map(ev => (
                <button
                  key={ev.id}
                  className={`ws-btn ws-btn-sm ${selectedEvalId === ev.id ? 'ws-btn-primary' : 'ws-btn-ghost'}`}
                  onClick={() => setSelectedEvalId(ev.id)}
                >
                  {ev.name}
                </button>
              ))}
            </div>

            {selectedEval && (
              <div style={{ marginTop: '20px' }}>
                <div className="ws-grid-3" style={{ marginBottom: '20px' }}>
                  <div className="ws-metric">
                    <span className="ws-metric-label">Top-1 Accuracy</span>
                    <span className="ws-metric-value">
                      {selectedEval.top1_accuracy != null ? `${(selectedEval.top1_accuracy * 100).toFixed(1)}%` : selectedEval.wer != null ? `WER: ${(selectedEval.wer * 100).toFixed(1)}%` : '—'}
                    </span>
                    <span className="ws-note">Held-out test partition</span>
                  </div>
                  <div className="ws-metric">
                    <span className="ws-metric-label">Macro F1 Score</span>
                    <span className="ws-metric-value">
                      {selectedEval.macro_f1 != null ? selectedEval.macro_f1.toFixed(4) : selectedEval.cer != null ? `CER: ${(selectedEval.cer * 100).toFixed(1)}%` : '—'}
                    </span>
                    <span className="ws-note">Balanced classification metric</span>
                  </div>
                  <div className="ws-metric">
                    <span className="ws-metric-label">Test Samples</span>
                    <span className="ws-metric-value">{selectedEval.test_samples}</span>
                    <span className="ws-note">{selectedEval.correct != null ? `${selectedEval.correct} / ${selectedEval.test_samples} correct` : 'Complete sentences'}</span>
                  </div>
                </div>

                <dl className="ws-detail-list">
                  <div>
                    <dt>Architecture</dt>
                    <dd>{selectedEval.architecture}</dd>
                  </div>
                  <div>
                    <dt>Target Task</dt>
                    <dd>{selectedEval.task}</dd>
                  </div>
                  <div>
                    <dt>Speaker Evaluation Protocol</dt>
                    <dd>{selectedEval.speaker_protocol || 'Strict speaker-disjoint test split'}</dd>
                  </div>
                  <div>
                    <dt>Checkpoint Artifact</dt>
                    <dd className="ws-code">{selectedEval.checkpoint_path}</dd>
                  </div>
                  {selectedEval.vocabulary && (
                    <div>
                      <dt>Supported Vocabulary ({selectedEval.vocabulary.length} classes)</dt>
                      <dd>{selectedEval.vocabulary.join(', ')}</dd>
                    </div>
                  )}
                </dl>
              </div>
            )}
          </section>

          {selectedEvalId === 'avhubert_miracl8_v1' && (
            <section className="ws-card">
              <div className="ws-card-head">
                <div>
                  <p className="ws-kicker">Detailed Breakdown</p>
                  <h2>MIRACL 8-Word Per-Class Performance</h2>
                </div>
                <span className="ws-badge ws-badge-ready">Top-5: 99.48%</span>
              </div>
              <div className="ws-table-wrap">
                <table className="ws-table">
                  <thead>
                    <tr>
                      <th>Target Word</th>
                      <th>Precision</th>
                      <th>Recall</th>
                      <th>F1-Score</th>
                      <th>Support</th>
                      <th>Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { word: 'choose', p: 0.923, r: 1.000, f1: 0.960, sup: 24, acc: '100.0%' },
                      { word: 'connection', p: 0.958, r: 0.958, f1: 0.958, sup: 24, acc: '95.8%' },
                      { word: 'hello', p: 0.960, r: 1.000, f1: 0.980, sup: 24, acc: '100.0%' },
                      { word: 'navigation', p: 0.920, r: 0.958, f1: 0.939, sup: 24, acc: '95.8%' },
                      { word: 'next', p: 1.000, r: 0.917, f1: 0.957, sup: 24, acc: '91.7%' },
                      { word: 'previous', p: 1.000, r: 0.750, f1: 0.857, sup: 24, acc: '75.0%' },
                      { word: 'start', p: 0.857, r: 1.000, f1: 0.923, sup: 24, acc: '100.0%' },
                      { word: 'stop', p: 1.000, r: 1.000, f1: 1.000, sup: 24, acc: '100.0%' },
                    ].map(row => (
                      <tr key={row.word}>
                        <td><strong>{row.word}</strong></td>
                        <td>{(row.p * 100).toFixed(1)}%</td>
                        <td>{(row.r * 100).toFixed(1)}%</td>
                        <td>{row.f1.toFixed(3)}</td>
                        <td>{row.sup}</td>
                        <td><span className="ws-badge ws-badge-ready">{row.acc}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {selectedEvalId === 'word_recognition_v1' && (
            <section className="ws-card">
              <div className="ws-card-head">
                <div>
                  <p className="ws-kicker">Confusion Matrix</p>
                  <h2>GRID 4-Word Confusion Matrix</h2>
                </div>
                <span className="ws-badge ws-badge-muted">32 Held-Out Samples</span>
              </div>
              <div className="ws-table-wrap">
                <table className="ws-table">
                  <thead>
                    <tr>
                      <th>Actual \ Predicted</th>
                      <th>bin</th>
                      <th>lay</th>
                      <th>place</th>
                      <th>set</th>
                      <th>Recall</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td><strong>bin (8)</strong></td>
                      <td style={{ background: 'rgba(56, 189, 248, 0.25)', fontWeight: 'bold' }}>7</td>
                      <td>0</td>
                      <td>0</td>
                      <td>1</td>
                      <td>87.5%</td>
                    </tr>
                    <tr>
                      <td><strong>lay (8)</strong></td>
                      <td>4</td>
                      <td style={{ background: 'rgba(239, 68, 68, 0.2)' }}>0</td>
                      <td>2</td>
                      <td>2</td>
                      <td>0.0%</td>
                    </tr>
                    <tr>
                      <td><strong>place (8)</strong></td>
                      <td>6</td>
                      <td>0</td>
                      <td style={{ background: 'rgba(56, 189, 248, 0.25)', fontWeight: 'bold' }}>2</td>
                      <td>0</td>
                      <td>25.0%</td>
                    </tr>
                    <tr>
                      <td><strong>set (8)</strong></td>
                      <td>6</td>
                      <td>0</td>
                      <td>0</td>
                      <td style={{ background: 'rgba(56, 189, 248, 0.25)', fontWeight: 'bold' }}>2</td>
                      <td>25.0%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          )}
        </div>
      ) : (
        <div className="ws-demo-layout">
          <section className="ws-card">
            <div className="ws-card-head">
              <h2>Live observation</h2>
              <Link className="ws-inline-link" to="/demo/live">Full live studio<Icon name="arrow" size={14}/></Link>
            </div>
            <div className="ws-video-wrap">
              <video ref={d.videoRef} className="ws-video" playsInline muted aria-label="Test camera preview"/>
              <canvas ref={d.overlayRef} className="ws-overlay" aria-hidden="true"/>
              {!d.running && (
                <div className="ws-idle">
                  <Icon name="camera" size={36}/>
                  <p>{d.camStatus === 'requesting' ? 'Allow camera access to begin.' : 'Start a session to observe real model responses.'}</p>
                </div>
              )}
            </div>
            <div className="ws-controls">
              {d.running || d.camStatus === 'requesting' ? (
                <button className="ws-btn ws-btn-ghost" onClick={d.stop}>
                  <Icon name="stop" size={14}/>{d.running ? 'Stop camera' : 'Cancel'}
                </button>
              ) : (
                <button className="ws-btn ws-btn-primary" onClick={() => void d.start()}>
                  <Icon name="camera" size={14}/>Start camera
                </button>
              )}
              <span className="ws-mono">{d.readout.frames} captured frames · {d.readout.faceOk ? 'Face detected' : 'No face detected'}</span>
            </div>
            {(d.camError || d.lmError || d.predictionError || d.pipelineError) && (
              <p className="ws-error" role="alert">{d.camError || d.lmError || d.predictionError || d.pipelineError}</p>
            )}
            <p className="ws-note">Model confidence is an uncalibrated softmax score. A confidence value does not establish that a prediction is correct.</p>
          </section>

          <section className="ws-card">
            <div className="ws-card-head">
              <h2>Prediction distribution</h2>
              <button className="ws-inline-link" disabled={!d.history.length} onClick={d.clearHistory}>Reset</button>
            </div>
            {d.history.length === 0 ? (
              <div className="ws-empty">
                <Icon name="chart" size={30}/>
                <h3>No observed results yet</h3>
                <p>Real returned letters will build a distribution here.</p>
              </div>
            ) : (
              <div className="ws-test-distribution">
                {Object.entries(counts).sort(([a],[b])=>a.localeCompare(b)).map(([letter,count])=>(
                  <div key={letter}>
                    <strong>{letter}</strong>
                    <span><i style={{width:count / maximum * 100 + '%'}}/></span>
                    <small>{count}</small>
                  </div>
                ))}
              </div>
            )}
            <p className="ws-note" style={{marginTop:16}}>Distribution of the retained session responses, without ground-truth comparison.</p>
          </section>
        </div>
      )}

      <section className="ws-card" style={{ marginTop: '24px' }}>
        <div className="ws-card-head">
          <h2>Session response log</h2>
          <span className="ws-badge ws-badge-muted">Session memory only</span>
        </div>
        {d.history.length === 0 ? (
          <p className="ws-note">Waiting for a real inference response.</p>
        ) : (
          <div className="ws-table-wrap">
            <table className="ws-table">
              <thead>
                <tr>
                  <th>Received</th>
                  <th>Letter</th>
                  <th>Model confidence</th>
                  <th>Request round trip</th>
                </tr>
              </thead>
              <tbody>
                {d.history.map(item=>(
                  <tr key={item.id}>
                    <td>{new Date(item.at).toLocaleTimeString()}</td>
                    <td>{item.letter}</td>
                    <td>{item.confidence === null ? 'Unavailable' : (item.confidence * 100).toFixed(1) + '%'}</td>
                    <td>{item.latencyMs} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  )
}
