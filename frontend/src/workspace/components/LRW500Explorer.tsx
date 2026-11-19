import { useState, useMemo } from 'react'
import { cyberAudio } from '../../utils/CyberSoundFX'
import { LRW_VOCABULARY, type LRWWordMeta } from '../../data/lrwVocabulary'

export function LRW500Explorer() {
  const [search, setSearch] = useState('')
  const [activeCluster, setActiveCluster] = useState<string>('ALL')
  const [selectedWord, setSelectedWord] = useState<LRWWordMeta>(LRW_VOCABULARY[0])

  const clusters = ['ALL', 'Bilabial', 'Labiodental', 'Alveolar', 'Velar', 'Vocalic']

  const filteredWords = useMemo(() => {
    return LRW_VOCABULARY.filter(item => {
      const matchSearch = item.word.toLowerCase().includes(search.toLowerCase().trim())
      const matchCluster = activeCluster === 'ALL' || item.cluster === activeCluster
      return matchSearch && matchCluster
    })
  }, [search, activeCluster])

  return (
    <section className="ws-card ws-lrw-explorer" aria-label="LRW-500 Vocabulary and Viseme Explorer" style={{ marginTop: '24px' }}>
      <div className="ws-card-head">
        <div>
          <p className="ws-kicker">Benchmark Corpus Explorer</p>
          <h2 style={{ fontSize: '18px', margin: '4px 0 0', letterSpacing: '-0.02em' }}>
            LRW-500 Vocabulary & Viseme Explorer
          </h2>
        </div>
        <span className="ws-badge ws-badge-ready">
          500 Target Classes · 29 Frames @ 25 FPS
        </span>
      </div>

      <p style={{ color: '#94a3b8', fontSize: '13px', lineHeight: 1.5, margin: '10px 0 16px' }}>
        Lip Reading in the Wild (LRW) provides 500 English word classes recorded under diverse head poses, lighting conditions, and speaker cadences.
        Viseme clusters group words by shared optical articulation dynamics.
      </p>

      {/* Search and Cluster Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center', marginBottom: '16px' }}>
        <div style={{ flex: '1 1 240px', position: 'relative' }}>
          <input
            type="search"
            className="ws-input"
            placeholder="Search word (e.g. ABOUT, PEOPLE, SYSTEM)..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            style={{ width: '100%', background: '#090c14', borderColor: '#1f2536' }}
          />
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
          {clusters.map(cluster => (
            <button
              key={cluster}
              type="button"
              className="ws-btn-tactile"
              onClick={() => {
                cyberAudio.playClick()
                setActiveCluster(cluster)
              }}
              style={{
                background: activeCluster === cluster ? '#182235' : '#121622',
                border: activeCluster === cluster ? '1px solid #38bdf8' : '1px solid #1f2536',
                color: activeCluster === cluster ? '#38bdf8' : '#94a3b8',
                borderRadius: '5px',
                padding: '4px 8px',
                fontSize: '11px',
                fontFamily: 'var(--font-mono)',
                fontWeight: activeCluster === cluster ? 600 : 400
              }}
            >
              {cluster.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Grid Layout: Word List + Detail Panel */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(280px, 1fr) 300px', gap: '16px' }}>
        {/* Words Grid */}
        <div style={{
          maxHeight: '340px',
          overflowY: 'auto',
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(130px, 1fr))',
          gap: '6px',
          padding: '8px',
          background: '#090c14',
          borderRadius: '8px',
          border: '1px solid #1a2233'
        }}>
          {filteredWords.map(item => {
            const isSelected = selectedWord.word === item.word
            return (
              <button
                key={item.word}
                type="button"
                className={`ws-btn-tactile ${isSelected ? 'is-selected' : ''}`}
                onClick={() => {
                  cyberAudio.playClick()
                  setSelectedWord(item)
                }}
                style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'flex-start',
                  padding: '8px 10px',
                  background: isSelected ? 'rgba(56, 189, 248, 0.12)' : '#10141f',
                  border: isSelected ? '1px solid #38bdf8' : '1px solid #1f2536',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  textAlign: 'left'
                }}
              >
                <strong style={{ color: isSelected ? '#38bdf8' : '#f1f5f9', fontSize: '12px', fontFamily: 'var(--font-mono)' }}>
                  {item.word}
                </strong>
                <span style={{ fontSize: '9px', color: '#64748b', marginTop: '2px' }}>
                  {item.cluster}
                </span>
              </button>
            )
          })}
        </div>

        {/* Word Detail Forensic Card */}
        <div style={{
          padding: '14px',
          background: '#10141f',
          border: '1px solid #20293d',
          borderRadius: '8px',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between'
        }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
              <h3 style={{ margin: 0, fontSize: '18px', color: '#38bdf8', fontFamily: 'var(--font-mono)', letterSpacing: '0.04em' }}>
                {selectedWord.word}
              </h3>
              <span style={{
                fontSize: '10px',
                background: 'rgba(16, 185, 129, 0.1)',
                color: '#10b981',
                border: '1px solid rgba(16, 185, 129, 0.25)',
                padding: '2px 6px',
                borderRadius: '4px',
                fontFamily: 'var(--font-mono)'
              }}>
                {selectedWord.cluster.toUpperCase()}
              </span>
            </div>

            <p style={{ color: '#94a3b8', fontSize: '12px', margin: '6px 0 12px', fontFamily: 'var(--font-mono)' }}>
              Phonetic IPA: <strong style={{ color: '#e2e8f0' }}>{selectedWord.phonemes}</strong>
            </p>

            <dl className="ws-detail-list" style={{ fontSize: '11px', margin: 0 }}>
              <div style={{ padding: '4px 0' }}>
                <dt style={{ color: '#64748b' }}>Temporal Duration</dt>
                <dd style={{ fontFamily: 'var(--font-mono)', color: '#f1f5f9' }}>{selectedWord.frames} frames (1.16s @ 25 Hz)</dd>
              </div>
              <div style={{ padding: '4px 0' }}>
                <dt style={{ color: '#64748b' }}>Training Clips</dt>
                <dd style={{ fontFamily: 'var(--font-mono)', color: '#10b981' }}>{selectedWord.trainClips} recordings</dd>
              </div>
              <div style={{ padding: '4px 0' }}>
                <dt style={{ color: '#64748b' }}>Validation Split</dt>
                <dd style={{ fontFamily: 'var(--font-mono)' }}>{selectedWord.valClips} clips</dd>
              </div>
              <div style={{ padding: '4px 0' }}>
                <dt style={{ color: '#64748b' }}>Held-out Test Split</dt>
                <dd style={{ fontFamily: 'var(--font-mono)' }}>{selectedWord.testClips} clips</dd>
              </div>
            </dl>
          </div>

          <div style={{
            marginTop: '12px',
            paddingTop: '8px',
            borderTop: '1px solid #1a2233',
            fontSize: '10px',
            color: '#64748b',
            lineHeight: 1.4
          }}>
            Evaluated via AV-HuBERT 103M encoder with Layer-9 representation extraction and temporal pooling.
          </div>
        </div>
      </div>
    </section>
  )
}
