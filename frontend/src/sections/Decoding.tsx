import { Reveal } from '../animations/Reveal'
import { SectionHead } from '../components/ui'
import { DECODER } from '../data/architecture'

/**
 * SECTION 07 — DECODING
 * CTC over the temporal representation. The sequence of frame predictions
 * (with blank states) collapses into the letter. Alphabet-scale only —
 * the current dataset contains spoken alphabet recordings, not sentences.
 */

// illustrative CTC collapse: blanks + repeats merge into a single letter
const CTC_ROW = ['∅', '∅', 'B', 'B', '∅', 'B', '∅', '∅']

export function Decoding() {
  return (
    <section className="section decoding" aria-labelledby="decoding-title">
      <div className="container">
        <SectionHead
          id="decoding-title"
          kicker="07 — Decoding"
          title={
            <>
              From movement <em>to meaning.</em>
            </>
          }
          lede="The system does not classify each frame independently. The whole temporal sequence is decoded together: repeated letters and blank states collapse into one prediction."
        />

        <div className="ctc-board">
          <Reveal className="ctc-seq" aria-label="Illustrative CTC frame sequence">
            <p className="mono-note">Per-frame sequence (illustrative)</p>
            <div className="ctc-frames">
              {CTC_ROW.map((s, i) => (
                <span key={i} className={`ctc-frame ${s === '∅' ? 'ctc-blank' : ''}`}>{s}</span>
              ))}
            </div>
          </Reveal>

          <Reveal className="ctc-connector" aria-hidden="true">
            <span className="ctc-line" />
            <p className="mono-note">CTC collapse</p>
            <span className="ctc-line" />
          </Reveal>

          <Reveal className="ctc-result">
            <span className="ctc-letter" aria-label="Decoded letter example">B</span>
            <p className="mono-note">One letter per spoken recording</p>
          </Reveal>
        </div>

        <Reveal className="ctc-vocab" aria-label="Output vocabulary">
          <p className="mono-note">Output vocabulary — {DECODER.classes} classes ({DECODER.blank} + A–Z)</p>
          <div className="ctc-letters">
            {DECODER.letters.map((l) => (
              <span key={l} className="ctc-vocab-letter">{l}</span>
            ))}
            <span className="ctc-vocab-letter ctc-vocab-letter--blank">{DECODER.blank}</span>
          </div>
        </Reveal>

        <Reveal as="p" className="mono-note ctc-note">
          Scope: spoken alphabet characters (A–Z). Continuous sentence recognition is not claimed.
        </Reveal>
      </div>
    </section>
  )
}
