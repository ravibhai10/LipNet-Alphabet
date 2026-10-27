import { useState } from 'react'
import { MathExpression } from '../../components/MathExpression'
import { EQUATIONS } from '../../lib/equations'

export function ImageFormationScene() {
  const [selectedPixel, setSelectedPixel] = useState<{ r: number; g: number; b: number; row: number; col: number }>({
    r: 184,
    g: 92,
    b: 104,
    row: 3,
    col: 4,
  })

  // 8x8 synthetic mouth lip patch matrix with realistic lip-skin tones
  const pixelPatch = [
    [ {r: 215, g: 170, b: 155}, {r: 210, g: 165, b: 150}, {r: 195, g: 140, b: 130}, {r: 180, g: 100, b: 105}, {r: 175, g: 95, b: 100}, {r: 182, g: 102, b: 106}, {r: 200, g: 145, b: 135}, {r: 215, g: 170, b: 155} ],
    [ {r: 210, g: 165, b: 150}, {r: 198, g: 145, b: 135}, {r: 182, g: 105, b: 110}, {r: 170, g: 85, b: 95},  {r: 168, g: 80, b: 90},  {r: 172, g: 88, b: 96},  {r: 185, g: 110, b: 115}, {r: 208, g: 162, b: 148} ],
    [ {r: 202, g: 150, b: 140}, {r: 185, g: 112, b: 118}, {r: 165, g: 78, b: 88},   {r: 150, g: 60, b: 72},  {r: 148, g: 58, b: 70},  {r: 152, g: 62, b: 74},  {r: 168, g: 82, b: 92},   {r: 198, g: 145, b: 135} ],
    [ {r: 190, g: 130, b: 125}, {r: 172, g: 92, b: 100},  {r: 152, g: 62, b: 75},   {r: 65, g: 25, b: 35},   {r: 60, g: 22, b: 32},   {r: 68, g: 28, b: 38},   {r: 154, g: 65, b: 78},   {r: 188, g: 128, b: 122} ],
    [ {r: 185, g: 122, b: 118}, {r: 168, g: 85, b: 95},   {r: 145, g: 55, b: 68},   {r: 45, g: 15, b: 22},   {r: 42, g: 12, b: 20},   {r: 48, g: 18, b: 24},   {r: 148, g: 58, b: 72},   {r: 184, g: 120, b: 115} ],
    [ {r: 192, g: 135, b: 128}, {r: 175, g: 95, b: 105},  {r: 158, g: 70, b: 82},   {r: 142, g: 52, b: 65},  {r: 140, g: 50, b: 62},  {r: 145, g: 55, b: 68},  {r: 162, g: 75, b: 88},   {r: 190, g: 132, b: 125} ],
    [ {r: 205, g: 155, b: 145}, {r: 190, g: 125, b: 122}, {r: 175, g: 95, b: 104},  {r: 165, g: 80, b: 92},  {r: 162, g: 78, b: 90},  {r: 168, g: 82, b: 95},  {r: 180, g: 105, b: 112}, {r: 202, g: 150, b: 140} ],
    [ {r: 215, g: 170, b: 155}, {r: 208, g: 160, b: 148}, {r: 195, g: 138, b: 130}, {r: 185, g: 115, b: 112}, {r: 182, g: 112, b: 110}, {r: 188, g: 118, b: 115}, {r: 198, g: 142, b: 132}, {r: 214, g: 168, b: 152} ],
  ]

  // Rec 601 Luma calculation
  const lumaY = Math.round(0.299 * selectedPixel.r + 0.587 * selectedPixel.g + 0.114 * selectedPixel.b)

  // Standard z-score normalization across typical batch (mean ~ 124.5, std ~ 48.2)
  const normX = ((lumaY - 124.5) / 48.2).toFixed(3)

  return (
    <section id="scene-pixels" className="science-scene" aria-label="Scene 2: From Light to Pixels">
      <div className="scene-header">
        <span className="scene-num">Scene 02 · Image Formation</span>
        <h2 className="scene-title">From Photons to Tensors.</h2>
        <p className="scene-lead">
          Visual lip-reading requires converting noisy web camera streams into standardized geometric coordinates. 
          Every photon passing through the sensor goes through a strict 5-stage transformation pipeline before entering the neural network.
        </p>
      </div>

      <div className="sci-split">
        {/* Left: 5-Stage Spatial Pipeline */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Spatial Preprocessing Pipeline</h3>
            <span className="sci-tag verified">LipNet Preprocessor</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {[
              {
                step: '01',
                title: 'RGB Video Frame (B, T, 3, H, W)',
                spec: 'Raw capture from OpenCV or MediaPipe at 640×480 px, 29–30 FPS.',
                detail: 'Captures full face environment with unconstrained lighting and background artifacts.'
              },
              {
                step: '02',
                title: '22-Landmark Oral Mesh Tracking',
                spec: 'MediaPipe Face Mesh indices [61, 146, 91, 181, 84, 17, 314, 405, 321...]',
                detail: 'Isolates vermilion border and inner mucous membrane contours on every frame.'
              },
              {
                step: '03',
                title: 'Affine Head-Tilt Canonicalization',
                spec: 'Calculates eye-mouth rotation angle θ, applies warpAffine rotation matrix.',
                detail: 'Eliminates false vertical lip displacement caused by head tilting during speech.'
              },
              {
                step: '04',
                title: 'Bounding Box Padding (dx: 30%, dy: 40%)',
                spec: 'x1 = min_x - 0.3·w, y1 = min_y - 0.4·h; clamped to boundary.',
                detail: 'Preserves chin drop and lateral lip stretching during extreme open vowels.'
              },
              {
                step: '05',
                title: 'Grayscale & Bilinear Resample (140 × 46 px)',
                spec: 'Downsamples to 140 height, 46 width, 1 channel tensor.',
                detail: 'Compresses spatial dimensions while retaining critical vertical aperture resolution.'
              }
            ].map((p) => (
              <div
                key={p.step}
                style={{
                  background: 'rgba(237, 235, 228, 0.02)',
                  border: '1px solid var(--line, rgba(237, 235, 228, 0.08))',
                  borderRadius: '8px',
                  padding: '0.85rem 1rem',
                  display: 'flex',
                  gap: '1rem',
                  alignItems: 'flex-start'
                }}
              >
                <div style={{
                  fontFamily: 'var(--font-mono)',
                  fontSize: '0.75rem',
                  color: 'var(--accent)',
                  background: 'rgba(143, 223, 203, 0.1)',
                  padding: '2px 8px',
                  borderRadius: '4px',
                  marginTop: '2px'
                }}>
                  {p.step}
                </div>
                <div>
                  <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink)' }}>{p.title}</div>
                  <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.74rem', color: 'var(--accent)', margin: '2px 0' }}>
                    {p.spec}
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--ink-dim)' }}>
                    {p.detail}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Mathematical Normalization & Interactive Pixel Inspector */}
        <div className="sci-card">
          <div className="sci-card-header">
            <h3 className="sci-card-title">Interactive Pixel Inspector</h3>
            <span className="sci-tag">Hover Patch</span>
          </div>

          <p style={{ fontSize: '0.84rem', color: 'var(--ink-dim)', margin: '0 0 1rem' }}>
            Hover over the 8×8 oral mucosa pixel patch below to inspect raw trichromatic RGB photon counts, 
            their Rec. 601 luma conversion, and zero-mean standardization.
          </p>

          {/* 8x8 Visual Patch */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(8, 1fr)',
            gap: '4px',
            background: '#08080a',
            padding: '12px',
            borderRadius: '10px',
            border: '1px solid rgba(237, 235, 228, 0.1)',
            marginBottom: '1.5rem'
          }}>
            {pixelPatch.map((row, rIdx) =>
              row.map((px, cIdx) => {
                const isSelected = selectedPixel.row === rIdx && selectedPixel.col === cIdx
                return (
                  <div
                    key={`${rIdx}-${cIdx}`}
                    onMouseEnter={() => setSelectedPixel({ ...px, row: rIdx, col: cIdx })}
                    style={{
                      aspectRatio: '1',
                      backgroundColor: `rgb(${px.r}, ${px.g}, ${px.b})`,
                      borderRadius: '3px',
                      cursor: 'crosshair',
                      outline: isSelected ? '2px solid var(--accent, #8fdfcb)' : 'none',
                      boxShadow: isSelected ? '0 0 10px var(--accent)' : 'none',
                      transition: 'transform 0.1s ease',
                      transform: isSelected ? 'scale(1.15)' : 'none',
                      zIndex: isSelected ? 2 : 1
                    }}
                    title={`Pixel [${rIdx},${cIdx}]: R=${px.r} G=${px.g} B=${px.b}`}
                  />
                )
              })
            )}
          </div>

          {/* Mathematical Formula Box */}
          <div className="math-box">
            <div className="math-formula">
              <MathExpression tex={EQUATIONS.grayscale}/>
            </div>
            <p className="math-desc">
              ITU-R BT.601 perceptual luma standard. Weighting reflects human retinal rod/cone sensitivity (peak sensitivity at 555nm green).
            </p>
          </div>

          <div className="math-box" style={{ borderLeftColor: 'var(--info)' }}>
            <div className="math-formula" style={{ color: 'var(--info)' }}>
              <MathExpression tex={EQUATIONS.normalization}/>
            </div>
            <p className="math-desc">
              Zero-mean unit-variance clip normalization. Eliminates ambient brightness disparities between different subjects and rooms.
            </p>
          </div>

          {/* Real-time calculated inspector metrics */}
          <div style={{
            background: 'rgba(237, 235, 228, 0.03)',
            borderRadius: '8px',
            border: '1px solid var(--line)',
            padding: '1rem',
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '0.75rem',
            fontFamily: 'var(--font-mono)',
            textAlign: 'center'
          }}>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--ink-faint)' }}>RGB PHOTONS</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--ink)', fontWeight: 600 }}>
                [{selectedPixel.r}, {selectedPixel.g}, {selectedPixel.b}]
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--ink-faint)' }}>CALCULATED LUMA (Y)</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--accent)', fontWeight: 600 }}>
                {lumaY} / 255
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.7rem', color: 'var(--ink-faint)' }}>TENSOR VALUE (x')</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--info)', fontWeight: 600 }}>
                {normX}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
