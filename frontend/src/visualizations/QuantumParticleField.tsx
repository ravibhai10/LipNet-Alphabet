import { useEffect, useRef } from 'react'

interface QuantumParticleFieldProps {
  className?: string
  particleCount?: number
  accentColor?: string
  secondaryColor?: string
}

export function QuantumParticleField({
  className = '',
  particleCount = 38,
  accentColor = 'rgba(99, 102, 241, 0.45)',
  secondaryColor = 'rgba(56, 189, 248, 0.35)',
}: QuantumParticleFieldProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let animId = 0
    let width = 0
    let height = 0
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5)

    type Particle = {
      x: number
      y: number
      vx: number
      vy: number
      radius: number
      color: string
      alpha: number
    }

    const particles: Particle[] = []

    const resize = () => {
      const parent = canvas.parentElement
      width = parent ? parent.clientWidth : window.innerWidth
      height = parent ? parent.clientHeight : window.innerHeight
      canvas.width = Math.round(width * dpr)
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${width}px`
      canvas.style.height = `${height}px`
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }

    resize()
    window.addEventListener('resize', resize)

    for (let i = 0; i < particleCount; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        radius: Math.random() * 1.5 + 0.6,
        color: Math.random() > 0.5 ? accentColor : secondaryColor,
        alpha: Math.random() * 0.3 + 0.1,
      })
    }

    let mouseX = -1000
    let mouseY = -1000

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect()
      mouseX = e.clientX - rect.left
      mouseY = e.clientY - rect.top
    }

    window.addEventListener('mousemove', onMouseMove, { passive: true })

    const render = () => {
      animId = requestAnimationFrame(render)
      ctx.clearRect(0, 0, width, height)

      for (let i = 0; i < particles.length; i++) {
        const p = particles[i]
        p.x += p.vx
        p.y += p.vy

        if (p.x < 0) { p.x = 0; p.vx *= -1 }
        else if (p.x > width) { p.x = width; p.vx *= -1 }
        if (p.y < 0) { p.y = 0; p.vy *= -1 }
        else if (p.y > height) { p.y = height; p.vy *= -1 }

        const dx = mouseX - p.x
        const dy = mouseY - p.y
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 100 && dist > 0) {
          p.x -= (dx / dist) * 0.4
          p.y -= (dy / dist) * 0.4
        }

        ctx.beginPath()
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2)
        ctx.fillStyle = p.color
        ctx.fill()

        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j]
          const dist2 = Math.hypot(p.x - p2.x, p.y - p2.y)
          if (dist2 < 90) {
            ctx.beginPath()
            ctx.moveTo(p.x, p.y)
            ctx.lineTo(p2.x, p2.y)
            const lineAlpha = (1 - dist2 / 90) * 0.14
            ctx.strokeStyle = `rgba(99, 102, 241, ${lineAlpha})`
            ctx.lineWidth = 0.8
            ctx.stroke()
          }
        }
      }
    }

    render()

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', resize)
      window.removeEventListener('mousemove', onMouseMove)
    }
  }, [particleCount, accentColor, secondaryColor])

  return (
    <div
      className={`ws-quantum-field ${className}`}
      style={{
        position: 'absolute',
        inset: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
        zIndex: 0,
        opacity: 0.85,
      }}
      aria-hidden="true"
    >
      <canvas ref={canvasRef} style={{ display: 'block', width: '100%', height: '100%' }} />
    </div>
  )
}
