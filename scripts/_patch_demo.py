from pathlib import Path

path = Path("frontend/src/sections/LiveDemoSection.tsx")
content = path.read_text(encoding="utf-8")

target = '<aside className="demo-panel" aria-label="Live inference readout">'

addition = """
            {/* Mode Selector */}
            <div className="demo-panel-block" style={{ paddingBottom: '0.8rem', borderBottom: '1px solid rgba(255,255,255,0.08)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span className="mono-note" style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  Recognition Task
                </span>
                <span style={{
                  fontSize: '0.65rem',
                  fontFamily: 'monospace',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  background: d.recognitionMode === 'word' ? 'rgba(102, 252, 241, 0.15)' : 'rgba(255,255,255,0.08)',
                  color: d.recognitionMode === 'word' ? '#66fcf1' : '#c5c6c7'
                }}>
                  {d.recognitionMode === 'word' ? 'Track A: Whole Words' : 'Track A0: Alphabet'}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  type="button"
                  onClick={() => d.setRecognitionMode('alphabet')}
                  style={{
                    flex: 1,
                    padding: '6px 8px',
                    fontSize: '0.75rem',
                    borderRadius: '6px',
                    border: d.recognitionMode === 'alphabet' ? '1px solid #8fdfcb' : '1px solid rgba(255,255,255,0.1)',
                    background: d.recognitionMode === 'alphabet' ? 'rgba(143, 223, 203, 0.15)' : 'transparent',
                    color: d.recognitionMode === 'alphabet' ? '#ffffff' : '#888899',
                    cursor: 'pointer'
                  }}
                >
                  Alphabet (A–Z)
                </button>
                <button
                  type="button"
                  onClick={() => d.setRecognitionMode('word')}
                  style={{
                    flex: 1,
                    padding: '6px 8px',
                    fontSize: '0.75rem',
                    borderRadius: '6px',
                    border: d.recognitionMode === 'word' ? '1px solid #66fcf1' : '1px solid rgba(255,255,255,0.1)',
                    background: d.recognitionMode === 'word' ? 'rgba(102, 252, 241, 0.15)' : 'transparent',
                    color: d.recognitionMode === 'word' ? '#ffffff' : '#888899',
                    cursor: 'pointer'
                  }}
                >
                  Whole Words (WordVideoNet)
                </button>
              </div>
              {d.recognitionMode === 'word' && (
                <div style={{ marginTop: '0.5rem', fontSize: '0.7rem', color: '#888899', fontFamily: 'monospace' }}>
                  Vocabulary (Track A): <strong style={{ color: '#66fcf1' }}>bin, lay, place, set</strong>
                </div>
              )}
            </div>"""

if target in content and "Recognition Task" not in content:
    content = content.replace(target, target + "\n" + addition, 1)
    # Also adjust detected letter label
    content = content.replace('<p className="mono-note">Detected letter</p>', '<p className="mono-note">{d.recognitionMode === "word" ? "Detected Spoken Word (Track A)" : "Detected Letter (Track A0)"}</p>')
    path.write_text(content, encoding="utf-8")
    print("LIVE_DEMO_PATCH_SUCCESS")
else:
    print("LIVE_DEMO_ALREADY_PATCHED_OR_NOT_FOUND")
