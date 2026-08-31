import { useRef, type ReactNode, type ElementType } from 'react'
import { gsap } from '../lib/gsap'
import { useGsapScene } from './useGsapScene'

interface RevealProps {
  children: ReactNode
  as?: ElementType
  className?: string
  /** direction of the entrance */
  from?: 'up' | 'down' | 'left' | 'right' | 'none'
  delay?: number
  id?: string
  role?: string
  'aria-hidden'?: boolean | 'true' | 'false'
  'aria-label'?: string
}

const OFFSETS = {
  up: { y: 44 },
  down: { y: -44 },
  left: { x: 56 },
  right: { x: -56 },
  none: {},
} as const

/**
 * Scroll-reveal wrapper (transform/opacity only — no per-frame React state).
 * With reduced motion the content simply renders in place.
 */
export function Reveal({
  children,
  as: Tag = 'div',
  className,
  from = 'up',
  delay = 0,
  id,
  role,
  'aria-hidden': ariaHidden,
  'aria-label': ariaLabel,
}: RevealProps) {
  const scope = useRef<HTMLElement>(null)

  useGsapScene(scope, () => {
    const el = scope.current
    if (!el) return
    gsap.fromTo(
      el,
      { autoAlpha: 0, ...OFFSETS[from] },
      {
        autoAlpha: 1,
        x: 0,
        y: 0,
        duration: 1.1,
        delay,
        ease: 'power3.out',
        scrollTrigger: { trigger: el, start: 'top 86%', once: true },
      },
    )
  })

  return (
    <Tag
      ref={scope}
      className={className}
      id={id}
      role={role}
      aria-hidden={ariaHidden}
      aria-label={ariaLabel}
    >
      {children}
    </Tag>
  )
}
