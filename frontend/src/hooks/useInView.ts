import { useEffect, useRef, useState } from 'react'

/**
 * IntersectionObserver-based visibility hook.
 * Used to gate canvases / rAF loops so nothing runs offscreen.
 */
export function useInView<T extends Element>(
  options: IntersectionObserverInit = { threshold: 0.15 },
): [React.RefObject<T>, boolean] {
  const ref = useRef<T>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    if (typeof IntersectionObserver === 'undefined') {
      setInView(true)
      return
    }
    const io = new IntersectionObserver((entries) => {
      setInView(entries[0]?.isIntersecting ?? false)
    }, options)
    io.observe(el)
    return () => io.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return [ref, inView]
}
