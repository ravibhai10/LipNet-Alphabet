/**
 * GlobeMap.tsx — Wireframe 3D globe with pulsing markers.
 */
import { useEffect, useRef } from "react"
import * as THREE from "three"
import { THEME } from "../../lib/threejs.config"

export function GlobeMap({ markers = 8 }: { markers?: number }) {
  const mountRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, (mount.clientWidth || 400) / (mount.clientHeight || 300), 0.1, 100)
    camera.position.set(0, 0, 4)
    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true })
    renderer.setSize(mount.clientWidth || 400, mount.clientHeight || 300)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    mount.appendChild(renderer.domElement)

    const globeGeo = new THREE.SphereGeometry(1.2, 32, 32)
    const globeMat = new THREE.MeshBasicMaterial({ color: THEME.cyan, wireframe: true, transparent: true, opacity: 0.25 })
    const globe = new THREE.Mesh(globeGeo, globeMat)
    scene.add(globe)

    const dotGeo = new THREE.SphereGeometry(0.04, 8, 8)
    const dotMeshes: THREE.Mesh[] = []
    for (let i = 0; i < markers; i++) {
      const mat = new THREE.MeshBasicMaterial({ color: i % 2 === 0 ? THEME.cyan : THEME.violet })
      const dot = new THREE.Mesh(dotGeo, mat)
      const phi = Math.random() * Math.PI * 2
      const theta = Math.acos(2 * Math.random() - 1)
      dot.position.set(
        Math.sin(theta) * Math.cos(phi) * 1.21,
        Math.cos(theta) * 1.21,
        Math.sin(theta) * Math.sin(phi) * 1.21,
      )
      scene.add(dot)
      dotMeshes.push(dot)
    }

    let animId = 0
    const tick = () => {
      animId = requestAnimationFrame(tick)
      globe.rotation.y += 0.002
      dotMeshes.forEach((d, i) => {
        d.scale.setScalar(1 + 0.4 * Math.sin(performance.now() * 0.005 + i))
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
      globeGeo.dispose(); dotGeo.dispose()
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
    }
  }, [markers])

  return <div ref={mountRef} style={{ width: "100%", height: "100%" }} aria-label="Global prediction map" />
}
