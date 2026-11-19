import type { CSSProperties } from 'react'

const paths: Record<string, string> = {
  grid: 'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
  camera: 'M4 7h3l2-3h6l2 3h3a1 1 0 0 1 1 1v11a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V8a1 1 0 0 1 1-1Z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
  arrow: 'M5 12h14 M13 6l6 6-6 6',
  chevron: 'M9 5l7 7-7 7',
  activity: 'M2 12h5l3-8 4 16 3-8h5',
  refresh: 'M20 8a8 8 0 0 0-14-3L3 8 M3 3v5h5 M4 16a8 8 0 0 0 14 3l3-3 M21 21v-5h-5',
  shield: 'M12 3l8 3v6c0 5-8 9-8 9S4 17 4 12V6l8-3Z M8 12l3 3 5-6',
  scan: 'M8 3H3v5 M16 3h5v5 M21 16v5h-5 M3 16v5h5 M7 12h10',
  stop: 'M5 5h14v14H5z',
  play: 'M7 4l14 8-14 8V4Z',
  info: 'M12 11v6 M12 7h.01 M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0',
  check: 'M5 12l4 4L19 6',
  alert: 'M12 3l10 18H2L12 3Z M12 9v5 M12 17h.01',
  close: 'M5 5l14 14 M19 5 5 19',
  database: 'M21 6c0 2-4 3-9 3S3 8 3 6s4-3 9-3 9 1 9 3Z M3 6v6c0 2 4 3 9 3s9-1 9-3V6 M3 12v6c0 2 4 3 9 3s9-1 9-3v-6',
  layers: 'M12 3 2 8l10 5 10-5-10-5Z M2 12l10 5 10-5 M2 16l10 5 10-5',
  flask: 'M9 3h6 M10 3v7L4 19a1 1 0 0 0 1 2h14a1 1 0 0 0 1-2l-6-9V3 M7 15h10',
  chart: 'M3 3v18h18 M7 16v-4 M12 16V8 M17 16V5',
  pipeline: 'M3 3h6v6H3z M15 15h6v6h-6z M9 6h6a3 3 0 0 1 3 3v6 M3 18h7 M7 15l3 3-3 3',
  book: 'M12 5c-3-2-6-2-9-1v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Z M12 5v15',
  cpu: 'M6 6h12v12H6z M9 9h6v6H9z M9 2v4 M15 2v4 M9 18v4 M15 18v4 M2 9h4 M2 15h4 M18 9h4 M18 15h4',
  menu: 'M4 6h16 M4 12h16 M4 18h16',
  collapse: 'M3 3h18v18H3z M8 3v18 M16 8l-4 4 4 4',
  sun: 'M12 3v2 M12 19v2 M4.22 4.22l1.42 1.42 M18.36 18.36l1.42 1.42 M1 12h2 M21 12h2 M4.22 19.78l1.42-1.42 M18.36 5.64l1.42-1.42 M12 7a5 5 0 1 0 0 10 5 5 0 0 0 0-10Z',
  moon: 'M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z',
  download: 'M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4 M7 10l5 5 5-5 M12 15V3',
  copy: 'M8 4H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2 M16 4h2a2 2 0 0 1 2 2v4 M8 4V2a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2 M8 4h8',
  search: 'M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16z M21 21l-4.35-4.35',
  volume: 'M11 5L6 9H2v6h4l5 4V5z M19.07 4.93a10 10 0 0 1 0 14.14 M15.54 8.46a5 5 0 0 1 0 7.07',
  'volume-x': 'M11 5L6 9H2v6h4l5 4V5z M23 9l-6 6 M17 9l6 6',
  zap: 'M13 2L3 14h9l-1 8 10-12h-9l1-8z',
  sliders: 'M4 21v-7 M4 10V3 M12 21v-9 M12 8V3 M20 21v-5 M20 12V3 M1 14h6 M9 8h6 M17 16h6',
}
export function Icon({ name, size = 18, className, style }: { name: string; size?: number; className?: string; style?: CSSProperties }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className={className} style={style} aria-hidden="true"><path d={paths[name] ?? paths.activity} /></svg>
}
export function BrandMark() {
  return <svg className="ws-brand-mark" width="32" height="32" viewBox="0 0 32 32" fill="none" aria-hidden="true"><rect x=".5" y=".5" width="31" height="31" rx="10" stroke="currentColor" strokeOpacity=".3"/><path d="M7 16h3l3-7 5 14 3-7h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>
}
