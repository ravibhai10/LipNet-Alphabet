import { useEffect, useRef } from 'react'
import * as THREE from 'three'

interface Props {
  aperture?: number | null
  speaking?: boolean
  className?: string
  width?: number | string
  height?: number | string
}

export function CyberFaceWireframe({ aperture = 0, speaking = false, className = '', width = '100%', height = 260 }: Props) {
  const mountRef = useRef<HTMLDivElement>(null)
  const animRef = useRef<number>(0)
  const apertureRef = useRef<number>(aperture || 0)
  const speakingRef = useRef<boolean>(speaking)

  useEffect(() => {
    apertureRef.current = aperture ?? 0
    speakingRef.current = speaking
  }, [aperture, speaking])

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return

    const w = mount.clientWidth || 320
    const h = typeof height === 'number' ? height : 260

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 100)
    camera.position.set(0, 0, 4.2)

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(w, h)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    // Build anatomical lip contour spline
    const numPoints = 28
    const lipVertices = new Float32Array(numPoints * 2 * 3)
    const lipGeo = new THREE.BufferGeometry()
    lipGeo.setAttribute('position', new THREE.BufferAttribute(lipVertices, 3))

    const lipMat = new THREE.LineBasicMaterial({
      color: 0x6366f1,
      linewidth: 1.5,
      transparent: true,
      opacity: 0.85
    })

    const lipLines = new THREE.LineSegments(lipGeo, lipMat)
    scene.add(lipLines)

    // Landmark vertices
    const nodeGeo = new THREE.BufferGeometry()
    const nodePos = new Float32Array(numPoints * 3)
    nodeGeo.setAttribute('position', new THREE.BufferAttribute(nodePos, 3))
    const nodeMat = new THREE.PointsMaterial({
      color: 0x818cf8,
      size: 0.06,
      transparent: true,
      opacity: 0.8
    })
    const nodePoints = new THREE.Points(nodeGeo, nodeMat)
    scene.add(nodePoints)

    // Gyroscopic reference rings
    const ringGeo1 = new THREE.RingGeometry(1.6, 1.615, 48)
    const ringMat1 = new THREE.MeshBasicMaterial({ color: 0x6366f1, side: THREE.DoubleSide, transparent: true, opacity: 0.12 })
    const outerRing = new THREE.Mesh(ringGeo1, ringMat1)
    scene.add(outerRing)

    const ringGeo2 = new THREE.RingGeometry(1.3, 1.31, 36)
    const ringMat2 = new THREE.MeshBasicMaterial({ color: 0x94a3b8, side: THREE.DoubleSide, transparent: true, opacity: 0.1 })
    const innerRing = new THREE.Mesh(ringGeo2, ringMat2)
    scene.add(innerRing)

    // Mouse drag rotation
    let isDragging = false
    let prevMouseX = 0
    let prevMouseY = 0
    let targetRotX = 0
    let targetRotY = 0

    const onMouseDown = (e: MouseEvent) => {
      isDragging = true
      prevMouseX = e.clientX
      prevMouseY = e.clientY
    }
    const onMouseMove = (e: MouseEvent) => {
      if (!isDragging) return
      const dx = e.clientX - prevMouseX
      const dy = e.clientY - prevMouseY
      targetRotY += dx * 0.008
      targetRotX += dy * 0.008
      targetRotX = Math.max(-0.5, Math.min(0.5, targetRotX))
      prevMouseX = e.clientX
      prevMouseY = e.clientY
    }
    const onMouseUp = () => { isDragging = false }

    const dom = renderer.domElement
    dom.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)

    let clock = 0

    const renderLoop = () => {
      clock += 0.03
      animRef.current = requestAnimationFrame(renderLoop)

      const mar = apertureRef.current
      const isSpeaking = speakingRef.current

      // Smoothly interpolate rotation
      scene.rotation.y += (targetRotY - scene.rotation.y) * 0.1
      scene.rotation.x += (targetRotX - scene.rotation.x) * 0.1

      outerRing.rotation.z += 0.002
      innerRing.rotation.z -= 0.003

      // Base breathing scale
      const breath = Math.sin(clock * 1.5) * 0.03
      const openAmount = Math.max(0.08, mar * 2.2 + (isSpeaking ? breath : breath * 0.5))

      const v = lipGeo.attributes.position.array as Float32Array
      const np = nodeGeo.attributes.position.array as Float32Array
      let vIdx = 0
      let nIdx = 0

      const half = 14
      for (let i = 0; i < half; i++) {
        const t = (i / (half - 1)) * 2 - 1
        const x = t * 1.25
        const arch = Math.sin((t + 1) * Math.PI * 0.5)
        const bow = Math.cos(t * Math.PI) * 0.12
        const yTop = (arch * 0.35 + bow) * 0.8 + (openAmount * 0.4)
        const z = Math.cos(t * Math.PI * 0.5) * 0.25

        const yBottom = -(arch * 0.5) * 0.8 - (openAmount * 0.6)

        np[nIdx++] = x
        np[nIdx++] = yTop
        np[nIdx++] = z

        np[nIdx++] = x
        np[nIdx++] = yBottom
        np[nIdx++] = z

        if (i < half - 1) {
          const nextT = ((i + 1) / (half - 1)) * 2 - 1
          const nextX = nextT * 1.25
          const nextArch = Math.sin((nextT + 1) * Math.PI * 0.5)
          const nextBow = Math.cos(nextT * Math.PI) * 0.12
          const nextYTop = (nextArch * 0.35 + nextBow) * 0.8 + (openAmount * 0.4)
          const nextYBottom = -(nextArch * 0.5) * 0.8 - (openAmount * 0.6)
          const nextZ = Math.cos(nextT * Math.PI * 0.5) * 0.25

          v[vIdx++] = x; v[vIdx++] = yTop; v[vIdx++] = z
          v[vIdx++] = nextX; v[vIdx++] = nextYTop; v[vIdx++] = nextZ

          v[vIdx++] = x; v[vIdx++] = yBottom; v[vIdx++] = z
          v[vIdx++] = nextX; v[vIdx++] = nextYBottom; v[vIdx++] = nextZ

          if (i % 2 === 0) {
            v[vIdx++] = x; v[vIdx++] = yTop; v[vIdx++] = z
            v[vIdx++] = x; v[vIdx++] = yBottom; v[vIdx++] = z
          }
        }
      }

      lipGeo.attributes.position.needsUpdate = true
      nodeGeo.attributes.position.needsUpdate = true

      if (isSpeaking) {
        lipMat.color.setHex(0x10b981)
        nodeMat.color.setHex(0x34d399)
      } else {
        lipMat.color.setHex(0x6366f1)
        nodeMat.color.setHex(0x818cf8)
      }

      renderer.render(scene, camera)
    }

    renderLoop()

    const ro = new ResizeObserver(() => {
      if (mount.clientWidth) {
        const currentH = typeof height === 'number' ? height : 260
        camera.aspect = mount.clientWidth / currentH
        camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, currentH)
      }
    })
    ro.observe(mount)

    return () => {
      cancelAnimationFrame(animRef.current)
      ro.disconnect()
      dom.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
      renderer.dispose()
      lipGeo.dispose()
      lipMat.dispose()
      nodeGeo.dispose()
      nodeMat.dispose()
      ringGeo1.dispose()
      ringMat1.dispose()
      ringGeo2.dispose()
      ringMat2.dispose()
      if (mount.contains(renderer.domElement)) {
        mount.removeChild(renderer.domElement)
      }
    }
  }, [height])

  return (
    <div 
      className={`cyber-3d-lip-hud ws-hud-bracket ${className}`}
      style={{
        position: 'relative',
        width,
        height,
        background: 'linear-gradient(135deg, rgba(14, 18, 28, 0.9), rgba(9, 12, 20, 0.95))',
        borderRadius: '14px',
        border: '1px solid rgba(99, 102, 241, 0.25)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
        overflow: 'hidden'
      }}
    >
      <div 
        ref={mountRef} 
        style={{ width: '100%', height: '100%', cursor: 'grab' }}
        title="Interactive 3D Oral Articulator — Drag to rotate"
      />

      {/* Telemetry Tag Overlay */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '12px',
        fontFamily: 'var(--font-mono)',
        fontSize: '10px',
        color: 'var(--ws-text)',
        display: 'flex',
        alignItems: 'center',
        gap: '6px',
        pointerEvents: 'none',
        background: 'var(--ws-surface2)',
        padding: '3px 8px',
        borderRadius: '4px',
        border: '1px solid var(--ws-border)'
      }}>
        <span style={{
          width: '5px',
          height: '5px',
          borderRadius: '50%',
          background: speaking ? 'var(--ws-success)' : 'var(--ws-accent)'
        }} />
        <span>3D ORAL ARTICULATOR · MAR {(aperture ? (aperture * 100).toFixed(1) : '0.0')}%</span>
      </div>

      <div style={{
        position: 'absolute',
        bottom: '8px',
        right: '12px',
        fontFamily: 'var(--font-mono)',
        fontSize: '9px',
        color: 'var(--ws-text-dim)',
        pointerEvents: 'none'
      }}>
        DRAG TO ROTATE 3D MESH
      </div>
    </div>
  )
}
