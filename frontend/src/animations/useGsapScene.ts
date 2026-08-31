import { useLayoutEffect, type RefObject } from 'react'
import { gsap, ScrollTrigger } from '../lib/gsap'
import { useReducedMotion } from '../hooks/useReducedMotion'

/**
 * Runs a GSAP scene inside gsap.context() with automatic cleanup and
 * ScrollTrigger refresh on fonts/layout settle. When the user prefers
 * reduced motion the scene callback is NOT invoked (sections then render
 * their static fallback layout).
 */
export function useGsapScene(
  scope: RefObject<HTMLElement>,
  scene: (ctx: { reduced: boolean }) => void,
) {
  const reduced = useReducedMotion()

  useLayoutEffect(() => {
    if (reduced) return
    const ctx = gsap.context(() => scene({ reduced }), scope)
    // Ensure pinned/scrubbed triggers measure correctly after fonts load.
    const refresh = () => ScrollTrigger.refresh()
    if (document.fonts?.ready) {
      document.fonts.ready.then(refresh).catch(() => {})
    }
    const t = window.setTimeout(refresh, 400)
    return () => {
      window.clearTimeout(t)
      ctx.revert()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [reduced])
}
