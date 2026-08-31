import type { ReactNode } from 'react'

export function Kicker({ children }: { children: ReactNode }) {
  return <p className="kicker">{children}</p>
}

export function SectionHead({
  id,
  kicker,
  title,
  lede,
}: {
  id?: string
  kicker: string
  title: ReactNode
  lede?: ReactNode
}) {
  return (
    <header className="section-head" id={id}>
      <Kicker>{kicker}</Kicker>
      <h2 className="headline">{title}</h2>
      {lede ? <p className="lede">{lede}</p> : null}
    </header>
  )
}

export function SpecChip({ k, v }: { k: string; v: string }) {
  return (
    <div className="spec-chip">
      <span className="spec-key">{k}</span>
      <span className="spec-value">{v}</span>
    </div>
  )
}

export function Arrow({ className = '' }: { className?: string }) {
  return (
    <svg
      className={`btn-arrow ${className}`}
      width="14"
      height="14"
      viewBox="0 0 14 14"
      fill="none"
      aria-hidden="true"
    >
      <path d="M1 7h11M8 3l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  )
}
