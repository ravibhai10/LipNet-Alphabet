import { useEffect, useRef, useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { BrandMark, Icon } from '../workspace/Icon'

const PAGE_GROUPS = [
  { title: 'Experience', pages: [
    { to: '/', label: 'Research exhibits', note: 'The main research presentation', icon: 'activity' },
    { to: '/demo/live', label: 'Live demo', note: 'Open the full camera studio', icon: 'camera' },
    { to: '/science', label: 'Science Lab', note: 'Explore the science interactively', icon: 'flask' },
    { to: '/research', label: 'The story', note: 'From a silent face to meaning', icon: 'book' },
  ]},
  { title: 'Research tools', pages: [
    { to: '/demo/dataset', label: 'Dataset explorer', note: 'Recordings and data quality', icon: 'database' },
    { to: '/demo/training', label: 'Training', note: 'Prepare and follow a training run', icon: 'cpu' },
    { to: '/demo/testing', label: 'Evaluation & testing', note: 'Inspect actual model responses', icon: 'chart' },
    { to: '/demo/models', label: 'Model registry', note: 'Checkpoints and model readiness', icon: 'layers' },
  ]},
  { title: 'Workspace', pages: [
    { to: '/demo/overview', label: 'Overview', note: 'The local research environment', icon: 'grid' },
    { to: '/demo/experiments', label: 'Experiments', note: 'Saved runs and reported metrics', icon: 'flask' },
    { to: '/demo/pipeline', label: 'Pipeline & architecture', note: 'Follow each processing stage', icon: 'pipeline' },
    { to: '/demo/docs', label: 'Documentation', note: 'Setup, camera, and training guide', icon: 'book' },
  ]},
]

export function StoryNavigation() {
  const [open, setOpen] = useState(false)
  const header = useRef<HTMLElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const location = useLocation()
  useEffect(() => { setOpen(false) }, [location.pathname])
  useEffect(() => {
    if (!open) return
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setOpen(false); trigger.current?.focus() }
    }
    const outside = (event: PointerEvent) => {
      if (!header.current?.contains(event.target as Node)) setOpen(false)
    }
    document.addEventListener('keydown', escape)
    document.addEventListener('pointerdown', outside)
    return () => { document.removeEventListener('keydown', escape); document.removeEventListener('pointerdown', outside) }
  }, [open])
  return <header className="story-nav" ref={header}>
    <a href="#top" className="story-brand" aria-label="LipNet AI home"><BrandMark/><strong>LIPNET<span> / AI</span></strong></a>
    <nav aria-label="Story navigation">
      <a href="#method" className="story-nav-secondary">The method</a>
      <Link to="/" className="story-nav-secondary">Research</Link>
      <Link to="/science" className="story-nav-science"><Icon name="flask" size={15}/><span>Science<span className="story-nav-lab"> Lab</span></span></Link>
      <Link to="/demo/live" className="story-nav-demo">Live demo <Icon name="arrow" size={15}/></Link>
      <button ref={trigger} type="button" className={'story-pages-toggle' + (open ? ' is-open' : '')}
        aria-expanded={open} aria-controls="story-page-directory" aria-label={open ? 'Close all pages' : 'Open all pages'} onClick={() => setOpen(value => !value)}>
        <span>All pages</span><Icon name={open ? 'close' : 'menu'} size={19}/>
      </button>
    </nav>
    <div id="story-page-directory" className="story-page-directory" hidden={!open}>
      <div className="story-directory-heading"><p>EXPLORE LIPNET AI</p><span>One project. Every perspective.</span></div>
      <nav className="story-directory-groups" aria-label="All project pages">
        {PAGE_GROUPS.map(group => <div className="story-directory-group" key={group.title}>
          <p>{group.title}</p>{group.pages.map(page => <Link key={page.to} to={page.to} onClick={() => { setOpen(false); if (page.to === '/') window.scrollTo({ top: 0 }) }}>
            <Icon name={page.icon} size={17}/><span><strong>{page.label}</strong><small>{page.note}</small></span><Icon name="arrow" size={14}/>
          </Link>)}
        </div>)}
      </nav>
      <div className="story-directory-footer"><span>Charcoal. Ivory. A little mint.</span><a href="#demo" onClick={() => setOpen(false)}>Try the inline demo <Icon name="arrow" size={14}/></a></div>
    </div>
  </header>
}

export function ProjectDestinations() {
  return <nav className="story-destinations" aria-label="Explore project pages">{[
    { to: '/demo/live', icon: 'camera', title: 'Live Demo', note: 'Open the complete camera studio.' },
    { to: '/science', icon: 'flask', title: 'Science Lab', note: 'Explore the ideas through interactive scenes.' },
    { to: '/demo/overview', icon: 'grid', title: 'Research workspace', note: 'Dataset, training, models, and evaluation tools.' },
    { to: '/research', icon: 'activity', title: 'Animated research', note: 'Visit the original cinematic exhibits.' },
  ].map(page => <Link to={page.to} key={page.to}><Icon name={page.icon} size={22}/><strong>{page.title}</strong><span>{page.note}</span><Icon name="arrow" size={16}/></Link>)}</nav>
}
