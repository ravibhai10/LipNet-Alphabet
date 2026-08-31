/**
 * threejs.config.ts — Shared Three.js theme and helpers.
 * "Neon Noir Sci-Fi Lab" palette.
 */
import * as THREE from "three"

export const THEME = {
  bg: 0x08080c, cyan: 0x00f0ff, violet: 0x7b61ff, accent: 0x4ee4c8,
  surface: 0x0f0f14, surface2: 0x16161e, text: 0xd8d8e8, muted: 0x7070a0,
  dim: 0x404060, error: 0xe44e4e, success: 0x4ee4a4,
  warning: 0xe4c44e, processing: 0x6e9cf5,
} as const

export const CAMERA_DEFAULTS = { fov: 50, near: 0.1, far: 100, position: new THREE.Vector3(0, 0, 5) }

export function makeStandardMaterial(color: number, emissive?: number) {
  return new THREE.MeshStandardMaterial({
    color, emissive: emissive ?? color, emissiveIntensity: 0.2,
    metalness: 0.7, roughness: 0.3, transparent: true, opacity: 0.9,
  })
}
