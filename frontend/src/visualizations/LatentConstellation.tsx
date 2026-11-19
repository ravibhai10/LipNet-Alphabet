import { useEffect, useRef, useState, useMemo } from 'react'
import { LRW_VOCABULARY, type LRWWordMeta } from '../data/lrwVocabulary'

interface LatentConstellationProps {
  activeWord?: string | null
  onWordClick?: (word: string) => void
  height?: number
}

interface NodePoint {
  meta: LRWWordMeta
  x: number
  y: number
  cluster: string
  color: string
}

// Cluster centroids in 2D embedding space
const CLUSTER_CONFIG: Record<string, { cx: number; cy: number; color: string }> = {
  Bilabial: { cx: -200, cy: -80, color: '#F43F5E' },
  Labiodental: { cx: -120, cy: 160, color: '#F59E0B' },
  Alveolar: { cx: 20, cy: -160, color: '#38BDF8' },
  Velar: { cx: 180, cy: 120, color: '#A855F7' },
  Vocalic: { cx: 140, cy: -80, color: '#10B981' },
}

export function LatentConstellation({ activeWord, onWordClick, height = 480 }: LatentConstellationProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [search, setSearch] = useState('')
  const [selectedCluster, setSelectedCluster] = useState<string>('ALL')
  const [hoveredNode, setHoveredNode] = useState<NodePoint | null>(null)

  // Camera transform state: pan & zoom
  const transform = useRef({ x: 0, y: 0, scale: 1.0 })
  const isDragging = useRef(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const animFrameId = useRef<number | null>(null)
  const pulseRef = useRef(0)

  // Generate deterministic 2D projection for words
  const nodes = useMemo<NodePoint[]>(() => {
    return LRW_VOCABULARY.map((meta, i) => {
      const conf = CLUSTER_CONFIG[meta.cluster] ?? { cx: 0, cy: 0, color: '#94A3B8' }
      // Golden angle spiral distribution per cluster
      const angle = (i * 137.5 * Math.PI) / 180
      const radius = 25 + Math.sqrt(i % 15) * 28 + (i % 7) * 4
      const x = conf.cx + Math.cos(angle) * radius
      const y = conf.cy + Math.sin(angle) * radius
      return {
        meta,
        x,
        y,
        cluster: meta.cluster,
        color: conf.color,
      }
    })
  }, [])

  // Find active node
  const activeNode = useMemo(() => {
    if (!activeWord) return null
    return nodes.find(n => n.meta.word.toUpperCase() === activeWord.toUpperCase().trim()) ?? null
  }, [nodes, activeWord])

  // Top-5 nearest neighbors to active node
  const nearestNeighbors = useMemo(() => {
    if (!activeNode) return []
    return [...nodes]
      .filter(n => n.meta.word !== activeNode.meta.word && n.cluster === activeNode.cluster)
      .map(n => {
        const dx = n.x - activeNode.x
        const dy = n.y - activeNode.y
        return { node: n, distSq: dx * dx + dy * dy }
      })
      .sort((a, b) => a.distSq - b.distSq)
      .slice(0, 5)
      .map(item => item.node)
  }, [nodes, activeNode])

  // Center camera on active node when it changes
  useEffect(() => {
    if (activeNode) {
      transform.current.x = -activeNode.x * transform.current.scale
      transform.current.y = -activeNode.y * transform.current.scale
    }
  }, [activeNode])

  // Canvas render loop
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let running = true

    const render = () => {
      if (!running) return
      pulseRef.current = (pulseRef.current + 0.05) % (Math.PI * 2)

      const w = canvas.width
      const h = canvas.height
      ctx.clearRect(0, 0, w, h)

      // Background matte fill
      ctx.fillStyle = '#07080B'
      ctx.fillRect(0, 0, w, h)

      ctx.save()
      // Center origin + camera pan/zoom
      ctx.translate(w / 2 + transform.current.x, h / 2 + transform.current.y)
      ctx.scale(transform.current.scale, transform.current.scale)

      // Grid crosshairs
      ctx.strokeStyle = '#1F2536'
      ctx.lineWidth = 1 / transform.current.scale
      ctx.beginPath()
      ctx.moveTo(-1000, 0); ctx.lineTo(1000, 0)
      ctx.moveTo(0, -1000); ctx.lineTo(0, 1000)
      ctx.stroke()

      // Concentric distance reference rings
      ctx.strokeStyle = 'rgba(31, 37, 54, 0.4)'
      ctx.setLineDash([4, 6])
      for (const r of [150, 300, 450]) {
        ctx.beginPath()
        ctx.arc(0, 0, r, 0, Math.PI * 2)
        ctx.stroke()
      }
      ctx.setLineDash([])

      // Draw cluster centroids and hairline hulls
      Object.entries(CLUSTER_CONFIG).forEach(([name, c]) => {
        if (selectedCluster !== 'ALL' && selectedCluster !== name) return
        ctx.fillStyle = 'rgba(255, 255, 255, 0.02)'
        ctx.beginPath()
        ctx.arc(c.cx, c.cy, 90, 0, Math.PI * 2)
        ctx.fill()

        ctx.fillStyle = c.color
        ctx.font = '10px monospace'
        ctx.fillText(name.toUpperCase(), c.cx - 28, c.cy - 100)
      })

      // Draw lines between active node and its nearest neighbors
      if (activeNode) {
        nearestNeighbors.forEach(nbr => {
          ctx.strokeStyle = 'rgba(56, 189, 248, 0.35)'
          ctx.lineWidth = 1 / transform.current.scale
          ctx.beginPath()
          ctx.moveTo(activeNode.x, activeNode.y)
          ctx.lineTo(nbr.x, nbr.y)
          ctx.stroke()
        })

        // Draw line from cluster centroid to active node
        const centroid = CLUSTER_CONFIG[activeNode.cluster]
        if (centroid) {
          ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)'
          ctx.setLineDash([2, 4])
          ctx.beginPath()
          ctx.moveTo(centroid.cx, centroid.cy)
          ctx.lineTo(activeNode.x, activeNode.y)
          ctx.stroke()
          ctx.setLineDash([])
        }
      }

      // Draw nodes
      const filterLower = search.toLowerCase().trim()
      nodes.forEach(node => {
        const matchesCluster = selectedCluster === 'ALL' || node.cluster === selectedCluster
        const matchesSearch = !filterLower || node.meta.word.toLowerCase().includes(filterLower)
        const isHovered = hoveredNode?.meta.word === node.meta.word
        const isActive = activeNode?.meta.word === node.meta.word
        const isNeighbor = nearestNeighbors.some(nbr => nbr.meta.word === node.meta.word)

        if (!matchesCluster) return

        let alpha = 0.5
        if (!matchesSearch) alpha = 0.1
        else if (isActive) alpha = 1.0
        else if (isNeighbor || isHovered) alpha = 0.9

        ctx.globalAlpha = alpha

        // Node dot
        const radius = isActive ? 6 : isNeighbor ? 4.5 : isHovered ? 5 : 3.5
        ctx.fillStyle = node.color
        ctx.beginPath()
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2)
        ctx.fill()

        // Active lock-on radar reticle
        if (isActive) {
          const pulse = Math.sin(pulseRef.current) * 4 + 14
          ctx.strokeStyle = '#38BDF8'
          ctx.lineWidth = 1.5 / transform.current.scale
          ctx.beginPath()
          ctx.arc(node.x, node.y, pulse, 0, Math.PI * 2)
          ctx.stroke()

          // Crosshairs
          ctx.beginPath()
          ctx.moveTo(node.x - pulse - 4, node.y); ctx.lineTo(node.x - pulse + 2, node.y)
          ctx.moveTo(node.x + pulse - 2, node.y); ctx.lineTo(node.x + pulse + 4, node.y)
          ctx.moveTo(node.x, node.y - pulse - 4); ctx.lineTo(node.x, node.y - pulse + 2)
          ctx.moveTo(node.x, node.y + pulse - 2); ctx.lineTo(node.x, node.y + pulse + 4)
          ctx.stroke()
        }

        // Labels
        if (isActive || isHovered || isNeighbor || (transform.current.scale > 1.4 && matchesSearch)) {
          ctx.font = `${isActive ? 'bold 11px' : '9px'} monospace`
          ctx.fillStyle = isActive ? '#FFFFFF' : node.color
          ctx.fillText(node.meta.word, node.x + 8, node.y + 3)
        }

        ctx.globalAlpha = 1.0
      })

      ctx.restore()
      animFrameId.current = requestAnimationFrame(render)
    }

    render()
    return () => {
      running = false
      if (animFrameId.current) cancelAnimationFrame(animFrameId.current)
    }
  }, [nodes, search, selectedCluster, hoveredNode, activeNode, nearestNeighbors])

  // Mouse pan & zoom handlers
  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault()
    const zoomFactor = e.deltaY < 0 ? 1.12 : 0.89
    const newScale = Math.min(3.5, Math.max(0.4, transform.current.scale * zoomFactor))
    transform.current.scale = newScale
  }

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    isDragging.current = true
    dragStart.current = { x: e.clientX - transform.current.x, y: e.clientY - transform.current.y }
  }

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return

    if (isDragging.current) {
      transform.current.x = e.clientX - dragStart.current.x
      transform.current.y = e.clientY - dragStart.current.y
      return
    }

    // Hit test for hover
    const rect = canvas.getBoundingClientRect()
    const mx = (e.clientX - rect.left - canvas.width / 2 - transform.current.x) / transform.current.scale
    const my = (e.clientY - rect.top - canvas.height / 2 - transform.current.y) / transform.current.scale

    let closest: NodePoint | null = null
    let minDist = 12 / transform.current.scale

    for (const node of nodes) {
      if (selectedCluster !== 'ALL' && node.cluster !== selectedCluster) continue
      const d = Math.hypot(node.x - mx, node.y - my)
      if (d < minDist) {
        minDist = d
        closest = node
      }
    }
    setHoveredNode(closest)
  }

  const handleMouseUp = () => {
    isDragging.current = false
  }

  const handleClick = () => {
    if (hoveredNode && onWordClick) {
      onWordClick(hoveredNode.meta.word)
    }
  }

  const resetCamera = () => {
    transform.current = { x: 0, y: 0, scale: 1.0 }
  }

  return (
    <div style={{
      background: '#0D1019',
      border: '1px solid #1F2536',
      borderRadius: '8px',
      overflow: 'hidden',
      fontFamily: 'monospace',
      position: 'relative',
    }}>
      {/* Header Controls */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '10px 14px',
        borderBottom: '1px solid #1F2536',
        background: '#121622',
        gap: '12px',
        flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ color: '#38BDF8', fontSize: '13px' }}>🌌</span>
          <span style={{ fontWeight: 600, fontSize: '12px', color: '#F1F5F9' }}>
            500-Word Latent Constellation
          </span>
          <span style={{ fontSize: '10px', color: '#64748B' }}>
            (Phonetic & Viseme Embeddings)
          </span>
        </div>

        {/* Cluster Filter Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          {['ALL', 'Bilabial', 'Labiodental', 'Alveolar', 'Velar', 'Vocalic'].map(cls => (
            <button
              key={cls}
              type="button"
              onClick={() => setSelectedCluster(cls)}
              style={{
                fontSize: '10px',
                padding: '2px 8px',
                borderRadius: '3px',
                border: '1px solid',
                borderColor: selectedCluster === cls ? '#38BDF8' : '#1F2536',
                background: selectedCluster === cls ? '#161D2D' : '#0D1019',
                color: selectedCluster === cls ? '#38BDF8' : '#94A3B8',
                cursor: 'pointer',
              }}
            >
              {cls}
            </button>
          ))}
        </div>

        {/* Search & Reset */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <input
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search word..."
            style={{
              padding: '3px 8px',
              fontSize: '11px',
              background: '#07080B',
              border: '1px solid #1F2536',
              borderRadius: '4px',
              color: '#F1F5F9',
              outline: 'none',
              width: '110px',
            }}
          />
          <button
            type="button"
            onClick={resetCamera}
            title="Reset Pan & Zoom"
            style={{
              padding: '3px 6px',
              fontSize: '10px',
              background: '#0D1019',
              border: '1px solid #1F2536',
              borderRadius: '4px',
              color: '#64748B',
              cursor: 'pointer',
            }}
          >
            Reset View
          </button>
        </div>
      </div>

      {/* Main Interactive Canvas */}
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onClick={handleClick}
        style={{
          display: 'block',
          width: '100%',
          height: `${height}px`,
          cursor: isDragging.current ? 'grabbing' : hoveredNode ? 'pointer' : 'crosshair',
        }}
      />

      {/* Hover / Active Telemetry HUD Overlay */}
      {(hoveredNode || activeNode) && (
        <div style={{
          position: 'absolute',
          bottom: '12px',
          left: '12px',
          background: 'rgba(13, 16, 25, 0.94)',
          border: '1px solid #1F2536',
          borderRadius: '6px',
          padding: '8px 12px',
          fontSize: '11px',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          pointerEvents: 'none',
          backdropFilter: 'blur(4px)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{
              color: (hoveredNode ?? activeNode)?.color,
              fontWeight: 700,
              fontSize: '13px',
              letterSpacing: '0.04em',
            }}>
              {(hoveredNode ?? activeNode)?.meta.word}
            </span>
            <span style={{
              background: '#1F2536',
              padding: '1px 6px',
              borderRadius: '3px',
              fontSize: '9px',
              color: '#94A3B8',
            }}>
              {(hoveredNode ?? activeNode)?.cluster}
            </span>
          </div>
          <div style={{ color: '#64748B', fontSize: '10px' }}>
            Phonemes: <span style={{ color: '#F1F5F9' }}>{(hoveredNode ?? activeNode)?.meta.phonemes}</span> · Frames: <span style={{ color: '#F1F5F9' }}>29</span>
          </div>
          {activeNode && (
            <div style={{ color: '#38BDF8', fontSize: '9px' }}>
              Target Locked · Top Nearest Neighbors: {nearestNeighbors.map(n => n.meta.word).join(', ')}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
