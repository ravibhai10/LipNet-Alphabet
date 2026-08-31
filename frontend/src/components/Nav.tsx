import { useEffect, useRef, useState } from 'react'

const LINKS = [
  { label: 'Research', href: '#research' },
  { label: 'Architecture', href: '#architecture' },
  { label: 'Data', href: '#data' },
  { label: 'Live Demo', href: '#demo' },
  { label: 'Results', href: '#results' },
  { label: 'About', href: '#about' },
]

/**
 * Fixed navigation. Transparent over the hero, compacts and frosts after
 * scroll. Uses a passive rAF-throttled scroll listener that only toggles a
 * class — no React state per frame.
 */
export function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const ticking = useRef(false)

  useEffect(() => {
    const onScroll = () => {
      if (ticking.current) return
      ticking.current = true
      requestAnimationFrame(() => {
        setScrolled(window.scrollY > 40)
        ticking.current = false
      })
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [open])

  return (
    <header className={`nav ${scrolled ? 'nav--scrolled' : ''} ${open ? 'nav--open' : ''}`}>
      <div className="nav-inner">
        <a className="nav-brand" href="#top" aria-label="Visual Speech AI — home">
          <span className="nav-brand-mark" aria-hidden="true" />
          VISUAL SPEECH AI
        </a>

        <nav className="nav-links" aria-label="Primary">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} onClick={() => setOpen(false)}>
              {l.label}
            </a>
          ))}
        </nav>

        <a className="btn btn-ghost nav-cta" href="/demo" onClick={() => setOpen(false)}>
          Try Live Demo
        </a>

        <button
          className="nav-toggle"
          aria-expanded={open}
          aria-controls="mobile-menu"
          aria-label={open ? 'Close menu' : 'Open menu'}
          onClick={() => setOpen((v) => !v)}
        >
          <span />
          <span />
        </button>
      </div>

      <div id="mobile-menu" className="nav-mobile" role="dialog" aria-modal="true" aria-label="Menu" hidden={!open}>
        <nav aria-label="Mobile">
          {LINKS.map((l, i) => (
            <a key={l.href} href={l.href} style={{ transitionDelay: `${60 + i * 40}ms` }} onClick={() => setOpen(false)}>
              {l.label}
            </a>
          ))}
        </nav>
        <a className="btn btn-solid" href="/demo" onClick={() => setOpen(false)}>
          Try Live Demo
        </a>
      </div>
    </header>
  )
}
