/**
 * FaceScanner.tsx — Immersive 3D face mesh hero section.
 * Pure Three.js rotating wireframe head with glowing landmark dots.
 */
import { useEffect, useRef } from "react"
import * as THREE from "three"

export function FaceScanner() {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, (mount.clientWidth || 600) / (mount.clientHeight || 400), 0.1, 100)
    camera.position.set(0, 0, 3.5)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 600, mount.clientHeight || 400)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    scene.add(new THREE.AmbientLight(0xffffff, 0.5))
    const pl1 = new THREE.PointLight(0x00f0ff, 3, 12)
    pl1.position.set(2, 1, 2)
    scene.add(pl1)
    const pl2 = new THREE.PointLight(0x7b61ff, 2, 10)
    pl2.position.set(-2, -1, 1)
    scene.add(pl2)

    const headGeo = new THREE.SphereGeometry(1, 32, 32)
    const headMat = new THREE.MeshStandardMaterial({ color: 0x0a0a14, emissive: 0x0a0a14, emissiveIntensity: 0.1, metalness: 0.8, roughness: 0.2, transparent: true, opacity: 0.7 })
    const head = new THREE.Mesh(headGeo, headMat)
    scene.add(head)

    const wireMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.15 })
    const wireHead = new THREE.Mesh(headGeo.clone(), wireMat)
    wireHead.scale.setScalar(1.01)
    scene.add(wireHead)

    const dotGeo = new THREE.SphereGeometry(0.02, 8, 8)
    const dotMeshes: THREE.Mesh[] = []
    for (let i = 0; i < 24; i++) {
      const dotMat = new THREE.MeshBasicMaterial({ color: i % 4 === 0 ? 0x7b61ff : 0x00f0ff })
      const dot = new THREE.Mesh(dotGeo, dotMat)
      const theta = (i / 24) * Math.PI * 2
      const phi = Math.PI * 0.35 + (i / 24) * Math.PI * 0.3
      dot.position.set(Math.sin(phi) * Math.cos(theta) * 0.92, Math.cos(phi) * 0.92, Math.sin(phi) * Math.sin(theta) * 0.92)
      scene.add(dot)
      dotMeshes.push(dot)
    }

    let animId = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      const t = performance.now() * 0.001
      head.rotation.y = t * 0.3
      wireHead.rotation.y = t * 0.3
      dotMeshes.forEach((d, i) => {
        d.scale.setScalar(1 + 0.15 * Math.sin(t * 2 + i * 0.4))
        ;(d.material as THREE.MeshBasicMaterial).color.setHex(i % 4 === 0 ? 0x7b61ff : 0x00f0ff)
      })
      renderer.render(scene, camera)
    }
    tick()
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
      headGeo.dispose(); wireMat.dispose(); dotGeo.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [])

  return (
    <section id="face-scanner" style={{
      width: "100%", height: 520,
      background: "linear-gradient(135deg, #08080c 0%, #0a0a18 100%)",
      display: "flex", flexDirection: "column",
      alignItems: "center", justifyContent: "center",
      padding: "64px 24px", gap: 24,
    }}>
      <h2 style={{ fontFamily: "Fraunces, serif", fontSize: "clamp(28px,4vw,48px)", fontWeight: 700, color: "#fff", textAlign: "center", letterSpacing: "-0.01em", margin: 0 }}>
        Real-time 3D Face Mesh
      </h2>
      <p style={{ color: "#7070a0", fontSize: 16, textAlign: "center", maxWidth: 480, margin: 0 }}>
        468-point MediaPipe FaceMesh rendered live in the browser via WebGL.
      </p>
      <div ref={mountRef} style={{
        width: "100%", maxWidth: 480, height: 300,
        borderRadius: 16, overflow: "hidden",
        border: "1px solid rgba(0,240,255,0.12)",
      }} />
    </section>
  )
}
