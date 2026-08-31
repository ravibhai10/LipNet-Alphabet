/**
 * ConformerExplorer.tsx — Interactive 3D Conformer block visualization.
 * Pure Three.js. Click a block to explore its sub-units.
 */
import { useEffect, useRef } from "react"
import * as THREE from "three"
import { CONFORMER } from "../data/architecture"

export function ConformerExplorer() {
  const mountRef = useRef<HTMLDivElement>(null)
  const selRef = useRef(-1)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, (mount.clientWidth || 600) / (mount.clientHeight || 400), 0.1, 100)
    camera.position.set(0, 2, 7)
    camera.lookAt(0, 0, 0)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 600, mount.clientHeight || 400)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.4))
    const dir = new THREE.DirectionalLight(0xffffff, 1.2)
    dir.position.set(3, 5, 3)
    scene.add(dir)
    const pL = new THREE.PointLight(0x00f0ff, 2, 15)
    pL.position.set(-3, 2, 2)
    scene.add(pL)

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

    let animId = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      const t = performance.now() * 0.001
      blocks.forEach((b, i) => {
        b.position.y = Math.sin(t * 0.4 + i * 0.3) * 0.05
        b.rotation.y = t * 0.1 + i * 0.1
        const mat = b.material as THREE.MeshStandardMaterial
        const isSel = i === selRef.current
        mat.emissiveIntensity = isSel ? 0.5 + 0.2 * Math.sin(t * 4) : 0.2
        b.scale.setScalar(isSel ? 1.1 : 1.0)
      })
      renderer.render(scene, camera)
    }
    tick()

    const handleClick = () => {
      const n = CONFORMER.blocks
      selRef.current = (selRef.current + 1) % n
    }
    mount.addEventListener("click", handleClick)

    const ro = new ResizeObserver(() => {
      if (mount.clientWidth && mount.clientHeight) {
        camera.aspect = mount.clientWidth / mount.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, mount.clientHeight)
      }
    })
    ro.observe(mount)

    return () => {
      cancelAnimationFrame(animId); ro.disconnect(); renderer.dispose()
      blockGeo.dispose(); blocks.forEach((b) => (b.material as THREE.Material).dispose())
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [])

  return (
    <section id="conformer-explorer" style={{
      width: "100%", background: "linear-gradient(180deg, #0a0a14 0%, #08080c 100%)",
      padding: "64px 24px", display: "flex", flexDirection: "column", alignItems: "center", gap: 32,
    }}>
      <h2 style={{ fontFamily: "Fraunces, serif", fontSize: "clamp(24px,3vw,40px)", fontWeight: 700, color: "#fff", margin: 0 }}>
        Conformer Architecture Explorer
      </h2>
      <p style={{ color: "#7070a0", fontSize: 15, textAlign: "center", maxWidth: 480 }}>
        {CONFORMER.blocks}× Conformer blocks × d={CONFORMER.dModel}, {CONFORMER.heads} heads. Click the canvas to cycle through blocks.
      </p>
      <div ref={mountRef} style={{ width: "100%", maxWidth: 640, height: 300, borderRadius: 16, overflow: "hidden", border: "1px solid rgba(0,240,255,0.12)" }} />
    </section>
  )
}
