
/**
 * FaceMesh.tsx — 3D face mesh overlay using pure Three.js
 * Projects MediaPipe FaceMesh landmarks (468 pts) into 3D space as
 * glowing cyan dots with an orbit animation.
 */
import { useEffect, useRef } from 'react'
import * as THREE from 'three'

interface FaceMeshProps {
  /** {x,y} in [0,1] normalized video coords */
  landmarks?: { x: number; y: number }[]
}

export function FaceMesh({ landmarks }: FaceMeshProps) {
  const mountRef = useRef<HTMLDivElement>(null)
  const posRef = useRef<Float32Array | null>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, (mount.clientWidth || 400) / (mount.clientHeight || 300), 0.1, 100)
    camera.position.set(0, 0, 5)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 400, mount.clientHeight || 300)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)
    scene.add(new THREE.AmbientLight(0xffffff, 0.6))
    const p1 = new THREE.PointLight(0x00f0ff, 2, 10); p1.position.set(2, 2, 3); scene.add(p1)
    const p2 = new THREE.PointLight(0x7b61ff, 1.5, 10); p2.position.set(-2, -1, 2); scene.add(p2)
    const MAX_PTS = 468
    const pos = new Float32Array(MAX_PTS * 3)
    posRef.current = pos
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setDrawRange(0, 0)
    const mat = new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.04, sizeAttenuation: true, transparent: true, opacity: 0.85 })
    const matGlow = new THREE.PointsMaterial({ color: 0x00f0ff, size: 0.09, sizeAttenuation: true, transparent: true, opacity: 0.2 })
    const points = new THREE.Points(geo, mat)
    const pointsGlow = new THREE.Points(geo, matGlow)
    scene.add(points); scene.add(pointsGlow)
    let animId = 0
    const animate = () => { animId = requestAnimationFrame(animate); points.rotation.y += 0.003; pointsGlow.rotation.y += 0.003; renderer.render(scene, camera) }
    animate()
    const ro = new ResizeObserver(() => {
      if (mount.clientWidth && mount.clientHeight) {
        camera.aspect = mount.clientWidth / mount.clientHeight; camera.updateProjectionMatrix()
        renderer.setSize(mount.clientWidth, mount.clientHeight)
      }
    })
    ro.observe(mount)
    return () => {
      cancelAnimationFrame(animId); ro.disconnect(); renderer.dispose(); geo.dispose(); mat.dispose(); matGlow.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [])

  useEffect(() => {
    const pos = posRef.current
    if (!pos || !landmarks) return
    landmarks.forEach((lm, i) => {
      if (i >= 468) return
      pos[i * 3] = (lm.x - 0.5) * 3
      pos[i * 3 + 1] = -(lm.y - 0.5) * 2
      pos[i * 3 + 2] = (Math.random() - 0.5) * 0.3
    })
    const mount = mountRef.current
    if (!mount) return
    const canvas = mount.querySelector('canvas') as HTMLCanvasElement | null
    if (!canvas) return
    const geo = (canvas as HTMLCanvasElement & { __three_geo?: THREE.BufferGeometry }).__three_geo
    if (geo) { geo.setDrawRange(0, landmarks.length); geo.attributes.position.needsUpdate = true }
  }, [landmarks])

  return <div ref={mountRef} style={{ width: '100%', height: '100%', background: 'transparent', borderRadius: 'inherit' }} aria-hidden='true' />
}
