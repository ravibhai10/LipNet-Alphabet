import { useState, useEffect } from 'react'
import { SectionHead } from '../components/ui'

export function FaceScanner() {
  const [activeFrame, setActiveFrame] = useState(1)
  const [isScanning, setIsScanning] = useState(true)
  const [scanY, setScanY] = useState(40)
  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null)

  // Animated laser sweep effect
  useEffect(() => {
    if (!isScanning) return
    const interval = setInterval(() => {
      setScanY((prev) => (prev >= 230 ? 40 : prev + 2))
    }, 30)
    return () => clearInterval(interval)
  }, [isScanning])

  // Landmark coordinates for 16 canonical lip points
  const lipLandmarks = [
    { id: 61, x: 120, y: 193, label: 'L_COMMISSURE' },
    { id: 146, x: 136, y: 185, label: 'L_UPPER_OUTER' },
    { id: 91, x: 156, y: 182, label: 'UPPER_CREST_L' },
    { id: 181, x: 176, y: 182, label: 'PHILTRUM_CENTER' },
    { id: 84, x: 196, y: 185, label: 'UPPER_CREST_R' },
    { id: 17, x: 216, y: 193, label: 'R_COMMISSURE' },
    { id: 314, x: 200, y: 206, label: 'LOWER_OUTER_R' },
    { id: 405, x: 176, y: 212, label: 'LOWER_VERMILION' },
    { id: 321, x: 152, y: 210, label: 'LOWER_OUTER_L' },
    { id: 78, x: 134, y: 193, label: 'INNER_L_COMM' },
    { id: 95, x: 156, y: 191, label: 'INNER_UPPER_L' },
    { id: 88, x: 176, y: 191, label: 'INNER_UPPER_MID' },
    { id: 178, x: 196, y: 193, label: 'INNER_R_COMM' },
    { id: 87, x: 188, y: 199, label: 'INNER_LOWER_R' },
    { id: 14, x: 176, y: 201, label: 'INNER_LOWER_MID' },
    { id: 317, x: 148, y: 199, label: 'INNER_LOWER_L' },
  ]

  // Frame data for timeline sequence
  const frames = [
    { id: 0, label: 'Frame t-1', aperture: '14.2%', h: 18, desc: 'Bilabial Onset' },
    { id: 1, label: 'Frame t (Active)', aperture: '64.8%', h: 36, desc: 'Vocalic Peak' },
    { id: 2, label: 'Frame t+1', aperture: '38.5%', h: 25, desc: 'Constriction' },
    { id: 3, label: 'Frame t+2', aperture: '12.0%', h: 16, desc: 'Terminal Bilabial' },
  ]

  return (
    <section id="face-scanner" className="section research-detail" aria-label="Visual Signal Analysis">
      <div className="container">
        <SectionHead
          kicker="01 · THE OPTICAL KINEMATIC SIGNAL"
          title={
            <>
              A face is full of <em>dynamic motion.</em>
            </>
          }
          lede="Mouth geometry, vermilion border displacement, and optical temporal velocities provide complementary silent speech signals. Hover over landmarks and scrub frames to inspect extraction coordinates."
        />

        <div className="research-schematic" style={{
          background: 'linear-gradient(180deg, #121217 0%, #090a0d 100%)',
          border: '1px solid rgba(143, 223, 203, 0.2)',
          boxShadow: '0 12px 40px rgba(0, 0, 0, 0.5)',
          borderRadius: '20px',
          padding: 'clamp(1.5rem, 3vw, 2.5rem)',
          position: 'relative',
          overflow: 'hidden'
        }}>
          {/* Top HUD Telemetry Bar */}
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderBottom: '1px solid rgba(237, 235, 228, 0.08)',
            paddingBottom: '1rem',
            marginBottom: '1.5rem',
            flexWrap: 'wrap',
            gap: '0.8rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.78rem'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <span style={{
                display: 'inline-block',
                width: '8px',
                height: '8px',
                borderRadius: '50%',
                background: 'var(--accent, #8fdfcb)',
                boxShadow: '0 0 10px var(--accent)'
              }} />
              <strong style={{ color: 'var(--ink)' }}>SCANNER HUD: OPTICAL LOCK ACTIVE</strong>
            </div>

            <div style={{ display: 'flex', gap: '1.2rem', color: 'var(--ink-dim)' }}>
              <span>MESH: <strong style={{ color: 'var(--accent)' }}>468 PTS</strong></span>
              <span>LIP ROI: <strong style={{ color: 'var(--info)' }}>140×46 PX</strong></span>
              <span>SAMPLING: <strong style={{ color: 'var(--ink)' }}>30 FPS</strong></span>
            </div>

            <button
              onClick={() => setIsScanning(!isScanning)}
              style={{
                background: isScanning ? 'rgba(143, 223, 203, 0.12)' : 'rgba(237, 235, 228, 0.05)',
                border: `1px solid ${isScanning ? 'var(--accent)' : 'var(--line)'}`,
                color: isScanning ? 'var(--accent)' : 'var(--ink-dim)',
                borderRadius: '999px',
                padding: '4px 12px',
                fontSize: '0.72rem',
                cursor: 'pointer',
                fontFamily: 'var(--font-mono)'
              }}
            >
              {isScanning ? '● Auto Scan: ON' : '○ Auto Scan: PAUSED'}
            </button>
          </div>

          {/* SVG Animated Face Mesh & Extraction Diagram */}
          <div style={{ width: '100%', overflowX: 'auto', textAlign: 'center' }}>
            <svg
              viewBox="0 0 740 310"
              fill="none"
              role="img"
              aria-label="Interactive Facial Landmark and Extraction Diagram"
              style={{ width: '100%', maxWidth: '850px', height: 'auto', margin: '0 auto' }}
            >
              <defs>
                {/* Laser scan line gradient */}
                <linearGradient id="scanBeam" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="transparent" />
                  <stop offset="50%" stopColor="var(--accent)" stopOpacity="0.8" />
                  <stop offset="100%" stopColor="transparent" />
                </linearGradient>

                {/* Laser glow filter */}
                <filter id="glowEffect" x="-20%" y="-20%" width="140%" height="140%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feComposite in="SourceGraphic" in2="blur" operator="over" />
                </filter>
              </defs>

              {/* Face Boundary Box */}
              <rect x="20" y="24" width="290" height="260" rx="20" stroke="rgba(237, 235, 228, 0.12)" fill="#07080a" />

              {/* Grid Background Lines inside Face Box */}
              <line x1="20" y1="90" x2="310" y2="90" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />
              <line x1="20" y1="160" x2="310" y2="160" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />
              <line x1="20" y1="230" x2="310" y2="230" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />
              <line x1="90" y1="24" x2="90" y2="284" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />
              <line x1="165" y1="24" x2="165" y2="284" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />
              <line x1="240" y1="24" x2="240" y2="284" stroke="rgba(237, 235, 228, 0.03)" strokeWidth="1" />

              {/* Facial Jaw Oval Contour */}
              <path
                d="M 92 105 C 92 50, 238 50, 238 105 V 162 C 238 230, 165 255, 165 255 C 165 255, 92 230, 92 162 Z"
                stroke="rgba(143, 223, 203, 0.25)"
                strokeWidth="1.5"
                fill="none"
              />

              {/* Eyes & Nose Bridge Wireframe */}
              <path d="M 125 120 H 145 M 185 120 H 205 M 165 125 V 158 H 155" stroke="rgba(237, 235, 228, 0.25)" strokeWidth="1.2" />

              {/* Active Laser Scan Beam */}
              {isScanning && (
                <g>
                  <line
                    x1="22"
                    y1={scanY}
                    x2="308"
                    y2={scanY}
                    stroke="url(#scanBeam)"
                    strokeWidth="2.5"
                    filter="url(#glowEffect)"
                  />
                  <polygon
                    points={`22,${scanY - 6} 308,${scanY - 6} 308,${scanY} 22,${scanY}`}
                    fill="rgba(143, 223, 203, 0.06)"
                  />
                </g>
              )}

              {/* Canonical Mouth Bounding Box with Margin Corners */}
              <rect
                x="110"
                y="172"
                width="114"
                height="48"
                rx="8"
                stroke="var(--accent)"
                strokeWidth="1.5"
                strokeDasharray="4 4"
                fill="rgba(143, 223, 203, 0.04)"
              />

              {/* Lip Contour Outer Path */}
              <path
                d="M 120 193 C 136 182, 196 182, 216 193 C 200 216, 140 216, 120 193 Z"
                stroke="var(--accent)"
                strokeWidth="1.6"
                fill="rgba(143, 223, 203, 0.08)"
              />

              {/* Lip Contour Inner Path (Dynamic Aperture based on active frame) */}
              <ellipse
                cx="168"
                cy="194"
                rx="30"
                ry={frames[activeFrame].id === 0 ? 3 : frames[activeFrame].id === 1 ? 12 : frames[activeFrame].id === 2 ? 8 : 2}
                fill={frames[activeFrame].id === 0 ? 'transparent' : 'rgba(10, 10, 12, 0.7)'}
                stroke="rgba(143, 223, 203, 0.6)"
                strokeWidth="1.2"
              />

              {/* Interactive Landmark Dots */}
              {lipLandmarks.map((pt, idx) => {
                const isHovered = hoveredPoint === idx
                return (
                  <g key={pt.id} onMouseEnter={() => setHoveredPoint(idx)} onMouseLeave={() => setHoveredPoint(null)} style={{ cursor: 'pointer' }}>
                    <circle
                      cx={pt.x}
                      cy={pt.y}
                      r={isHovered ? 4.5 : 2.2}
                      fill={isHovered ? 'var(--info)' : 'var(--accent)'}
                      filter={isHovered ? 'url(#glowEffect)' : 'none'}
                    />
                  </g>
                )
              })}

              {/* Transformation Vector Arrow to Temporal Frames */}
              <g transform="translate(325, 145)">
                <path d="M 0 0 H 45" stroke="var(--accent)" strokeWidth="2" strokeDasharray="3 3" />
                <polygon points="45,-5 55,0 45,5" fill="var(--accent)" />
                <text x="25" y="-12" textAnchor="middle" fontSize="9" fill="var(--accent)" fontFamily="monospace">
                  CROP &amp; ALIGN
                </text>
              </g>

              {/* 4 Temporal Frames Breakdown */}
              {frames.map((f, i) => {
                const isSelected = activeFrame === f.id
                return (
                  <g
                    key={f.id}
                    transform={`translate(${395 + i * 85}, 85)`}
                    onClick={() => setActiveFrame(f.id)}
                    style={{ cursor: 'pointer' }}
                  >
                    <rect
                      width="75"
                      height="115"
                      rx="10"
                      fill={isSelected ? 'rgba(143, 223, 203, 0.1)' : 'rgba(237, 235, 228, 0.02)'}
                      stroke={isSelected ? 'var(--accent)' : 'rgba(237, 235, 228, 0.15)'}
                      strokeWidth={isSelected ? 2 : 1}
                    />

                    {/* Simulated Mouth Crop in Frame */}
                    <g transform="translate(10, 30)">
                      <rect width="55" height="50" rx="4" fill="#060608" stroke="rgba(237, 235, 228, 0.08)" />
                      <ellipse
                        cx="27.5"
                        cy="25"
                        rx="20"
                        ry={f.h / 3}
                        fill="none"
                        stroke={isSelected ? 'var(--accent)' : 'rgba(237, 235, 228, 0.4)'}
                        strokeWidth="1.4"
                      />
                      <ellipse
                        cx="27.5"
                        cy="25"
                        rx="12"
                        ry={Math.max(1, f.h / 6)}
                        fill={f.id === 0 ? 'transparent' : 'rgba(143, 223, 203, 0.2)'}
                        stroke={isSelected ? 'rgba(143, 223, 203, 0.6)' : 'rgba(237, 235, 228, 0.2)'}
                      />
                    </g>

                    <text x="37.5" y="100" textAnchor="middle" fontSize="10" fill={isSelected ? 'var(--ink)' : 'var(--ink-dim)'} fontFamily="monospace">
                      F_{i}
                    </text>
                    <text x="37.5" y="132" textAnchor="middle" fontSize="9" fill={isSelected ? 'var(--accent)' : 'var(--ink-faint)'} fontFamily="monospace">
                      {f.aperture}
                    </text>
                  </g>
                )
              })}
            </svg>
          </div>

          {/* Interactive Landmark Coordinates HUD Banner */}
          <div style={{
            marginTop: '1.5rem',
            background: 'rgba(237, 235, 228, 0.02)',
            border: '1px solid var(--line)',
            borderRadius: '10px',
            padding: '1rem 1.4rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '1rem',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.8rem'
          }}>
            <div>
              {hoveredPoint !== null ? (
                <div>
                  <span style={{ color: 'var(--accent)' }}>LANDMARK #{lipLandmarks[hoveredPoint].id}: </span>
                  <strong style={{ color: 'var(--ink)' }}>{lipLandmarks[hoveredPoint].label}</strong>
                  <span style={{ color: 'var(--ink-dim)', marginLeft: '10px' }}>
                    [x: {lipLandmarks[hoveredPoint].x}, y: {lipLandmarks[hoveredPoint].y}]
                  </span>
                </div>
              ) : (
                <div style={{ color: 'var(--ink-dim)' }}>
                  Hover over any landmark dot on the facial diagram to inspect raw 3D mesh coordinates.
                </div>
              )}
            </div>

            <div style={{ display: 'flex', gap: '1.2rem' }}>
              <div>
                <span style={{ color: 'var(--ink-faint)' }}>ACTIVE STAGE: </span>
                <span style={{ color: 'var(--accent)' }}>{frames[activeFrame].desc} ({frames[activeFrame].aperture} Aperture)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
