/**
 * ParticleSystem.tsx — Glowing particle flow for visualisations.
 */
import { useEffect, useRef } from "react"
import * as THREE from "three"
import { THEME } from "../../lib/threejs.config"

export function ParticleSystem({ count = 200, color = THEME.cyan, speed = 0.5, bounds = 4 }: { count?: number; color?: number; speed?: number; bounds?: number }) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(50, (mount.clientWidth || 400) / (mount.clientHeight || 300), 0.1, 100)
    camera.position.set(0, 0, 6)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 400, mount.clientHeight || 300)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const positions = new Float32Array(count * 3)
    const velocities = new Float32Array(count * 3)
    for (let i = 0; i < count; i++) {
      positions[i * 3] = (Math.random() - 0.5) * bounds
      positions[i * 3 + 1] = (Math.random() - 0.5) * bounds
      positions[i * 3 + 2] = (Math.random() - 0.5) * bounds
      velocities[i * 3] = (Math.random() - 0.5) * speed
      velocities[i * 3 + 1] = (Math.random() - 0.5) * speed
      velocities[i * 3 + 2] = (Math.random() - 0.5) * speed
    }
    const geo = new THREE.BufferGeometry()
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3))
    const mat = new THREE.PointsMaterial({ color, size: 0.04, sizeAttenuation: true, transparent: true, opacity: 0.7, blending: THREE.AdditiveBlending })
    const points = new THREE.Points(geo, mat)
    scene.add(points)

    let animId = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      const pos = geo.attributes.position.array as Float32Array
      for (let i = 0; i < count; i++) {
        pos[i * 3] += velocities[i * 3] * 0.01
        pos[i * 3 + 1] += velocities[i * 3 + 1] * 0.01
        pos[i * 3 + 2] += velocities[i * 3 + 2] * 0.01
        if (Math.abs(pos[i * 3]) > bounds) velocities[i * 3] *= -1
        if (Math.abs(pos[i * 3 + 1]) > bounds) velocities[i * 3 + 1] *= -1
        if (Math.abs(pos[i * 3 + 2]) > bounds) velocities[i * 3 + 2] *= -1
      }
      geo.attributes.position.needsUpdate = true
      points.rotation.y += 0.001
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
      geo.dispose(); mat.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [count, color, speed, bounds])

  return <div ref={mountRef} style={{ width: "100%", height: "100%", pointerEvents: "none" }} aria-hidden="true" />
}
