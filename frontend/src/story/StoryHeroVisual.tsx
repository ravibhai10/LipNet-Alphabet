import { useState } from 'react'
import { HeroVisual } from '../visualizations/HeroVisual'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { Icon } from '../workspace/Icon'

/** The original animated face study, reused in the narrative hero. */
export function StoryHeroVisual() {
  const [paused, setPaused] = useState(false)
  const reduced = useReducedMotion()
  return <figure className="story-hero-art story-hero-face" data-paused={paused || reduced}>
    <HeroVisual paused={paused}/>
    <div className="story-face-scan" aria-hidden="true"/>
    <span className="story-art-label story-art-label-top">THE VISIBLE PART OF SPEECH</span>
    <span className="story-art-coordinate">LANDMARKS / MOTION / TIME</span>
    <div className="story-art-footer"><figcaption>Illustration · no camera or inference running</figcaption>
      <button type="button" className="story-motion-toggle" disabled={reduced} onClick={() => setPaused(value => !value)}
        aria-label={reduced ? 'Reduced motion enabled' : paused ? 'Play face animation' : 'Pause face animation'}>
        <Icon name={paused ? 'play' : 'stop'} size={12}/>{reduced ? 'Reduced motion' : paused ? 'Play animation' : 'Pause animation'}
      </button>
    </div>
  </figure>
}
