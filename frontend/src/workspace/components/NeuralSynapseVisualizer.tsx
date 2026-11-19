import { useState, useEffect, useRef } from 'react'
import { cyberAudio } from '../../utils/CyberSoundFX'

interface Stage {
  id: string
  name: string
  code: string
  tensorShape: string
  details: string
  params: string
  accent: string
}

const STAGES: Stage[] = [
  {
    id: 'roi',
    name: 'Mouth ROI Video Stream',
    code: 'INPUT_FRAMES',
    tensorShape: '[B, 32, 1, 88, 88]',
    details: '32-frame stabilized temporal window cropped around landmarks [61, 291, 0, 17] at 25 FPS.',
    params: '0 (Preprocessed)',
    accent: '#64748b'
  },
  {
    id: 'conv3d',
    name: 'Spatiotemporal 3D-CNN',
    code: 'CONV3D_FRONTEND',
    tensorShape: '[B, 64, 16, 44, 44]',
    details: 'Spatiotemporal kernel (5x5x5) capturing phonetic lip-opening and closing acceleration.',
    params: '48.2 K',
    accent: '#6366f1'
  },
  {
    id: 'resnet',
    name: 'ResNet-18 Spatial Encoder',
    code: 'RESNET18_BACKBONE',
    tensorShape: '[B, 512, 16, 1, 1]',
    details: '4 residual stage blocks with 2D convolutions producing high-level visual oral embeddings.',
    params: '11.2 M',
    accent: '#818cf8'
  },
  {
    id: 'avhubert',
    name: 'AV-HuBERT Base Transformer',
    code: 'TRANSFORMER_L12',
    tensorShape: '[B, 16, 768]',
    details: '12 layers, 12 self-attention heads, d_model=768. Self-supervised visual speech representation.',
    params: '90.3 M',
    accent: '#a855f7'
  },
  {
    id: 'head',
    name: 'Classification Head',
    code: 'LINEAR_PROJECTION',
    tensorShape: '[B, 8]',
    details: 'Trained linear projection to 8 MIRACL target classes with temperature calibration (tau=1.2).',
    params: '6.1 K',
    accent: '#f59e0b'
  },
  {
    id: 'lexicon',
    name: 'Rejection & Consensus',
    code: 'POSTERIOR_EVAL',
    tensorShape: 'Word Label + Conf %',
    details: 'Two-window consensus buffer, out-of-vocabulary negative phrase rejection, zero fabrication.',
    params: 'Rule-Based Engine',
    accent: '#10b981'
  }
]

export function NeuralSynapseVisualizer() {
  const [selectedStage, setSelectedStage] = useState<Stage>(STAGES[3])
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animRef = useRef<number>(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let width = (canvas.width = canvas.parentElement?.clientWidth || 700)
    let height = (canvas.height = 120)

    const handleResize = () => {
      if (!canvas || !canvas.parentElement) return
      width = canvas.width = canvas.parentElement.clientWidth
      height = canvas.height = 120
    }
    window.addEventListener('resize', handleResize)

    interface Packet {
      stageIdx: number
      progress: number
      speed: number
      color: string
    }

    const packets: Packet[] = [
      { stageIdx: 0, progress: 0.1, speed: 0.015, color: '#64748b' },
      { stageIdx: 1, progress: 0.5, speed: 0.018, color: '#6366f1' },
      { stageIdx: 2, progress: 0.2, speed: 0.014, color: '#818cf8' },
      { stageIdx: 3, progress: 0.7, speed: 0.016, color: '#a855f7' },
      { stageIdx: 4, progress: 0.3, speed: 0.02, color: '#f59e0b' }
    ]

    const render = () => {
      animRef.current = requestAnimationFrame(render)
      ctx.clearRect(0, 0, width, height)

      const stageCount = STAGES.length
      const paddingX = 40
      const usableW = width - paddingX * 2
      const stepX = usableW / (stageCount - 1)
      const centerY = height * 0.45

      // Base connecting track
      ctx.beginPath()
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)'
      ctx.lineWidth = 2
      ctx.moveTo(paddingX, centerY)
      ctx.lineTo(width - paddingX, centerY)
      ctx.stroke()

      // Active pipeline beam
      const gradient = ctx.createLinearGradient(paddingX, centerY, width - paddingX, centerY)
      gradient.addColorStop(0, '#64748b')
      gradient.addColorStop(0.25, '#6366f1')
      gradient.addColorStop(0.5, '#818cf8')
      gradient.addColorStop(0.75, '#a855f7')
      gradient.addColorStop(1, '#10b981')

      ctx.beginPath()
      ctx.strokeStyle = gradient
      ctx.lineWidth = 1.5
      ctx.moveTo(paddingX, centerY)
      ctx.lineTo(width - paddingX, centerY)
      ctx.stroke()

      // Update and draw flowing packets
      packets.forEach(p => {
        p.progress += p.speed
        if (p.progress >= 1) {
          p.progress = 0
          p.stageIdx = (p.stageIdx + 1) % (stageCount - 1)
        }

        const startX = paddingX + p.stageIdx * stepX
        const currentX = startX + p.progress * stepX

        ctx.beginPath()
        ctx.arc(currentX, centerY, 3, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.fill()
      })

      // Draw Stage Nodes
      STAGES.forEach((stg, i) => {
        const x = paddingX + i * stepX
        const isSelected = selectedStage.id === stg.id

        if (isSelected) {
          ctx.beginPath()
          ctx.arc(x, centerY, 11, 0, Math.PI * 2)
          ctx.strokeStyle = stg.accent
          ctx.lineWidth = 1.5
          ctx.stroke()
        }

        // Inner solid node
        ctx.beginPath()
        ctx.arc(x, centerY, isSelected ? 6 : 4.5, 0, Math.PI * 2)
        ctx.fillStyle = isSelected ? '#ffffff' : stg.accent
        ctx.fill()

        // Stage label below
        ctx.fillStyle = isSelected ? '#ffffff' : '#94a3b8'
        ctx.font = `${isSelected ? '600 ' : '400 '}10px Inter, system-ui, sans-serif`
        ctx.textAlign = 'center'
        ctx.fillText(stg.code, x, centerY + 24)
      })
    }

    render()

    return () => {
      cancelAnimationFrame(animRef.current)
      window.removeEventListener('resize', handleResize)
    }
  }, [selectedStage])

  return (
    <div 
      className="neural-synapse-visualizer ws-card ws-hud-bracket"
      style={{
        padding: '22px',
        position: 'relative',
        background: 'linear-gradient(180deg, rgba(16, 20, 32, 0.88) 0%, rgba(9, 12, 20, 0.96) 100%)',
        borderRadius: '14px',
        border: '1px solid rgba(99, 102, 241, 0.25)',
        boxShadow: '0 12px 36px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        overflow: 'hidden'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--ws-accent)' }} />
          <h3 style={{ margin: 0, fontSize: '13px', fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--ws-text)' }}>
            Neural Pipeline Architecture
          </h3>
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '10px', color: 'var(--ws-text-dim)' }}>
          CLICK NODE TO INSPECT TENSOR
        </span>
      </div>

      {/* Synapse Canvas */}
      <div style={{ width: '100%', position: 'relative', cursor: 'pointer' }}>
        <canvas ref={canvasRef} style={{ width: '100%', height: '120px', display: 'block' }} />
      </div>

      {/* Clickable Stage Button Bar */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(95px, 1fr))',
        gap: '6px',
        marginTop: '8px'
      }}>
        {STAGES.map((stg) => {
          const isSelected = selectedStage.id === stg.id
          return (
            <button
              key={stg.id}
              onClick={() => {
                cyberAudio.playClick()
                setSelectedStage(stg)
              }}
              style={{
                background: isSelected ? 'var(--ws-surface2)' : 'transparent',
                border: isSelected ? `1px solid ${stg.accent}` : '1px solid var(--ws-border)',
                borderRadius: '6px',
                padding: '6px 8px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'border-color 0.15s ease'
              }}
            >
              <div style={{ fontSize: '10px', fontFamily: 'var(--font-mono)', color: isSelected ? stg.accent : 'var(--ws-text-muted)', fontWeight: 550 }}>
                {stg.code}
              </div>
            </button>
          )
        })}
      </div>

      {/* Selected Stage Inspector */}
      <div style={{
        marginTop: '14px',
        padding: '12px 16px',
        background: 'var(--ws-surface2)',
        borderRadius: '8px',
        border: '1px solid var(--ws-border)',
        borderLeft: `3px solid ${selectedStage.accent}`,
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '12px'
      }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '13px', fontWeight: 600, color: 'var(--ws-text)' }}>
              {selectedStage.name}
            </span>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: '10px',
              padding: '2px 6px',
              borderRadius: '4px',
              background: 'var(--ws-surface3)',
              color: selectedStage.accent
            }}>
              {selectedStage.tensorShape}
            </span>
          </div>
          <p style={{ margin: '4px 0 0', fontSize: '12px', color: 'var(--ws-text-muted)', maxWidth: '560px', lineHeight: 1.45 }}>
            {selectedStage.details}
          </p>
        </div>

        <div style={{
          textAlign: 'right',
          fontFamily: 'var(--font-mono)',
          fontSize: '11px',
          color: 'var(--ws-text)'
        }}>
          <span style={{ color: 'var(--ws-text-dim)' }}>Params: </span>
          <strong style={{ color: selectedStage.accent, fontVariantNumeric: 'tabular-nums' }}>{selectedStage.params}</strong>
        </div>
      </div>
    </div>
  )
}
