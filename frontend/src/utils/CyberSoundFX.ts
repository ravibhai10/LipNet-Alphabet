/**
 * CyberSoundFX.ts — Procedural Web Audio Sound Synthesizer
 * Zero external audio files or dependencies. Generates subtle, crisp,
 * tactile sci-fi micro-feedback directly via Web Audio API oscillators.
 */

class SoundEngine {
  private ctx: AudioContext | null = null
  private enabled: boolean = true

  constructor() {
    try {
      const stored = localStorage.getItem('lipnet.audio_enabled')
      this.enabled = stored !== null ? stored === 'true' : true
    } catch {
      this.enabled = true
    }
  }

  private initCtx(): AudioContext | null {
    if (typeof window === 'undefined') return null
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      if (AudioCtx) {
        this.ctx = new AudioCtx()
      }
    }
    if (this.ctx && this.ctx.state === 'suspended') {
      this.ctx.resume().catch(() => {})
    }
    return this.ctx
  }

  public isEnabled(): boolean {
    return this.enabled
  }

  public setEnabled(val: boolean) {
    this.enabled = val
    try {
      localStorage.setItem('lipnet.audio_enabled', String(val))
    } catch {
      /* ignore storage failure */
    }
  }

  public toggle(): boolean {
    this.setEnabled(!this.enabled)
    if (this.enabled) {
      this.playClick()
    }
    return this.enabled
  }

  /** Subtle UI hover tick (super soft, high frequency) */
  public playHover() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sine'
      osc.frequency.setValueAtTime(1400, now)
      osc.frequency.exponentialRampToValueAtTime(900, now + 0.015)

      gain.gain.setValueAtTime(0.015, now)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.015)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now)
      osc.stop(now + 0.016)
    } catch {
      /* ignore audio errors */
    }
  }

  /** Tactile button / switch click */
  public playClick() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'triangle'
      osc.frequency.setValueAtTime(950, now)
      osc.frequency.exponentialRampToValueAtTime(320, now + 0.035)

      gain.gain.setValueAtTime(0.04, now)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.035)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now)
      osc.stop(now + 0.036)
    } catch {
      /* ignore audio errors */
    }
  }

  /** Biometric target lock / mouth detected chirp */
  public playTargetLock() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const osc1 = ctx.createOscillator()
      const osc2 = ctx.createOscillator()
      const gain = ctx.createGain()

      osc1.type = 'sine'
      osc2.type = 'sine'

      osc1.frequency.setValueAtTime(659.25, now) // E5
      osc1.frequency.setValueAtTime(880, now + 0.04) // A5

      osc2.frequency.setValueAtTime(1318.5, now) // E6
      osc2.frequency.setValueAtTime(1760, now + 0.04) // A6

      gain.gain.setValueAtTime(0.035, now)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.12)

      osc1.connect(gain)
      osc2.connect(gain)
      gain.connect(ctx.destination)

      osc1.start(now)
      osc2.start(now)
      osc1.stop(now + 0.12)
      osc2.stop(now + 0.12)
    } catch {
      /* ignore */
    }
  }

  /** Prediction decoded holographic sweep */
  public playDecodeWord() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const freqs = [523.25, 659.25, 783.99, 1046.5] // C - E - G - C arpeggio
      freqs.forEach((freq, idx) => {
        const osc = ctx.createOscillator()
        const gain = ctx.createGain()
        const startTime = now + idx * 0.025
        const stopTime = startTime + 0.09

        osc.type = 'sine'
        osc.frequency.setValueAtTime(freq, startTime)

        gain.gain.setValueAtTime(0.03, startTime)
        gain.gain.exponentialRampToValueAtTime(0.0001, stopTime)

        osc.connect(gain)
        gain.connect(ctx.destination)

        osc.start(startTime)
        osc.stop(stopTime)
      })
    } catch {
      /* ignore */
    }
  }

  /** Command palette opened chime */
  public playPaletteOpen() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sine'
      osc.frequency.setValueAtTime(330, now)
      osc.frequency.exponentialRampToValueAtTime(880, now + 0.08)

      gain.gain.setValueAtTime(0.04, now)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.08)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now)
      osc.stop(now + 0.085)
    } catch {
      /* ignore */
    }
  }

  /** Damped low buzz on rejection or face loss */
  public playReject() {
    if (!this.enabled) return
    const ctx = this.initCtx()
    if (!ctx) return

    try {
      const now = ctx.currentTime
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sawtooth'
      osc.frequency.setValueAtTime(140, now)
      osc.frequency.exponentialRampToValueAtTime(80, now + 0.07)

      gain.gain.setValueAtTime(0.025, now)
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.07)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now)
      osc.stop(now + 0.075)
    } catch {
      /* ignore */
    }
  }
}

export const cyberAudio = new SoundEngine()
