/**
 * ConformerBlock.tsx — Interactive 3D Conformer block explorer
 * Pure Three.js. Click blocks to select; use MHSA/Conv/FFN buttons to
 * expand sub-unit detail. All values sourced from data/architecture.ts.
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { CONFORMER } from '../../data/architecture'

export function ConformerExplorer() {
  const mountRef = useRef<HTMLDivElement>(null)
  const selectedRef = useRef(0)
  const detailRef = useRef<string | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, 600 / 320, 0.1, 100)
    camera.position.set(0, 2, 7)
    camera.lookAt(0, 0, 0)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 600, mount.clientHeight || 320)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)
    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const dir = new THREE.DirectionalLight(0xffffff, 1.2)
    dir.position.set(3, 5, 3)
    scene.add(dir)
    const pLight = new THREE.PointLight(0x00f0ff, 2, 15)
    pLight.position.set(-3, 2, 2)
    scene.add(pLight)
    const blockGeo = new THREE.BoxGeometry(1.1, 0.7, 0.6)
    const spacing = 1.6
    const totalWidth = CONFORMER.blocks * spacing
    const startX = -(totalWidth - spacing) / 2
    const blocks: THREE.Mesh[] = []
    for (let i = 0; i < CONFORMER.blocks; i++) {
      const mat = new THREE.MeshStandardMaterial({ color: 0x4ee4c8, emissive: 0x4ee4c8, emissiveIntensity: 0.2, metalness: 0.7, roughness: 0.3, transparent: true, opacity: 0.85 })
      const mesh = new THREE.Mesh(blockGeo, mat)
      mesh.position.set(startX + i * spacing, 0, 0)
      mesh.userData.index = i
      scene.add(mesh)
      blocks.push(mesh)
      const wireMat = new THREE.LineBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.3 })
      scene.add(new THREE.LineSegments(new THREE.EdgesGeometry(blockGeo), wireMat))
    }
    const subColors = [0x00f0ff, 0x7b61ff, 0x4ee4c8]
    const subGeo = new THREE.BoxGeometry(0.35, 0.3, 0.3)
    const subBlocks: THREE.Mesh[] = subColors.map((color) => {
      const mat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.6, metalness: 0.5, roughness: 0.4 })
      const m = new THREE.Mesh(subGeo, mat)
      m.visible = false
      scene.add(m)
      return m
    })
    let animId = 0, t = 0
    const animate = () => {
      animId = requestAnimationFrame(animate)
      t += 0.016
      const sel = selectedRef.current
      const detail = detailRef.current
      const detailIdx = detail ? ['attention', 'conv', 'ffn'].indexOf(detail) : -1
      blocks.forEach((b, i) => {
        b.position.y = Math.sin(t * 0.4 + i * 0.3) * 0.05
        b.rotation.y = t * 0.1 + i * 0.1
        const mat = b.material as THREE.MeshStandardMaterial
        mat.emissiveIntensity = i === sel ? 0.5 + 0.2 * Math.sin(t * 4) : 0.2
        b.scale.setScalar(i === sel ? 1.1 : 1.0)
        if (i === sel && detailIdx >= 0) {
          b.getWorldPosition(subBlocks[detailIdx].position)
        }
      })
      renderer.render(scene, camera)
    }
    animate()
    const ro = new ResizeObserver(() => {
      if (mount.clientWidth && mount.clientHeight) {
        camera.aspect = mount.clientWidth / mount.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, mount.clientHeight)
      }
    })
    ro.observe(mount)
    return () => {
      cancelAnimationFrame(animId)
      ro.disconnect()
      renderer.dispose()
      blockGeo.dispose()
      subGeo.dispose()
      blocks.forEach((b) => (b.material as THREE.Material).dispose())
      subBlocks.forEach((b) => (b.material as THREE.Material).dispose())
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [])

  return (
    <div ref={mountRef}
      style={{ width: '100%', height: 280, cursor: 'pointer', background: 'transparent' }}
      onClick={(e) => {
        const n = CONFORMER.blocks
        const clickX = e.clientX - e.currentTarget.getBoundingClientRect().left
        const idx = Math.max(0, Math.min(n - 1, Math.floor((clickX / e.currentTarget.clientWidth) * n)))
        selectedRef.current = idx
      }}
      role="button" aria-label="Conformer block stack" />
  )
}