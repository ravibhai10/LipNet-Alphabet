const modes = [
  { mode: 'Isolated letters', size: '26 · A–Z', status: 'CPU execution verified', detail: 'The saved CNN–BiLSTM returns one letter distribution per clip through /api/predict. It does not perform CTC decoding. Webcam accuracy remains unvalidated.' },
  { mode: 'MIRACL whole words', size: '8 supported words', status: '182/192 held-out clips · 94.79%', detail: 'A frozen AV-HuBERT encoder and supervised word head recognize choose, connection, hello, navigation, next, previous, start and stop. Test speakers were excluded from training; prior pilot utterances were quarantined. This score does not measure webcam accuracy or apply to the 500-word model.' },
  { mode: 'LRW whole words', size: '500 ordered classes', status: 'Separate research API · benchmark pending', detail: 'A genuine publisher-trained ResNet18 + DC-TCN is connected at /api/predict/word/lrw500. Its ordered author vocabulary matches data/vocab.txt. Real public AFTERNOON input has passed inference; a small diagnostic is not a full LRW evaluation. MediaPipe alignment differs from the publisher detector. The legacy local 500-output CNN checkpoints remain unverified.' },
  { mode: 'AV-HuBERT alphabet', size: '768 features per frame', status: 'Encoder available · A–Z head absent', detail: 'The official encoder is used by the eight-word experiment. No complete supervised 26-letter AV-HuBERT head is verified; select the existing alphabet checkpoint for A–Z execution.' },
  { mode: 'Continuous sentences', size: 'GRID: 33 available speakers', status: 'Data available · decoder unavailable', detail: 'The GRID inventory has 32,891 usable video/annotation pairs. Its letter-slot and bounded word experiments are separate from sentence recognition. CTC utilities exist, but no compatible trained visual sentence decoder and tokenizer have been verified.' },
]

export function ModelCapabilitiesNotice() {
  return <aside className="science-capabilities" aria-label="Verified model capabilities" data-testid="model-capabilities">
    <div className="sci-card-header"><h2 className="sci-card-title">What the artifacts actually support</h2><span className="sci-tag">Research status · 29 Oct 2026</span></div>
    <p className="math-desc">An ordered vocabulary, a working inference request and a held-out recognition benchmark provide different evidence. Current model readiness is reported separately in the live workspace.</p>
    <div className="science-capability-grid">{modes.map(item => <article key={item.mode}>
      <div className="science-capability-heading"><h3>{item.mode}</h3><span>{item.size}</span></div>
      <p className="science-capability-status">{item.status}</p><p>{item.detail}</p>
    </article>)}</div>
    <p className="math-desc">Isolated-word classification does not establish continuous transcription. Confidence and unknown-word rejection remain uncalibrated. Evidence: <code>WORD_ACCURACY_IMPROVEMENT.md</code> and <code>EXPANDED_WORD_RECOGNITION.md</code>.</p>
  </aside>
}
