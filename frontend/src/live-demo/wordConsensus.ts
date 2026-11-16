/** Confirmation for the existing automatic-window research pipeline. */
export class WordConsensus {
  private candidate: string | null = null
  private utterance: number | null = null
  private windows = 0

  reset() {
    this.candidate = null
    this.utterance = null
    this.windows = 0
  }

  update(word: string | null, uncertain: boolean, utterance: number): boolean {
    if (!word || uncertain) {
      this.reset()
      return false
    }
    this.windows = (this.candidate === word && this.utterance === utterance) ? this.windows + 1 : 1
    this.candidate = word
    this.utterance = utterance
    return this.windows >= 2
  }
}
