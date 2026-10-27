import { useState } from 'react'
import { SectionHead } from '../components/ui'
const streams = [
  {id:'appearance',title:'Appearance',description:'Texture and shape of the visible mouth region.'},
  {id:'geometry',title:'Geometry',description:'Spatial relationships between measured facial landmarks.'},
  {id:'motion',title:'Motion',description:'Changes in the visual signal from one frame to the next.'},
]
export function AblationStudio() {
  const [selected,setSelected] = useState<string[]>(streams.map(item=>item.id))
  return <section id="ablation-studio" className="section research-detail"><div className="container"><SectionHead kicker="REPRESENTATION EXPLORER" title={<>Different signals.<br/><em>One research question.</em></>} lede="Explore what each representation contributes conceptually. Actual ablation metrics require recorded evaluations for each model configuration."/><div className="research-schematic"><div className="research-streams">{streams.map(item=><button className={selected.includes(item.id)?'is-active':''} key={item.id} aria-pressed={selected.includes(item.id)} onClick={()=>setSelected(values=>values.includes(item.id)?values.filter(id=>id!==item.id):[...values,item.id])}><span>{item.title}</span><p>{item.description}</p><small>{selected.includes(item.id)?'Selected for inspection':'Select to inspect'}</small></button>)}</div><p className="research-body">{selected.length ? selected.length + ' conceptual streams selected.' : 'Select a representation to inspect its role.'}</p><p className="mono-note">Architecture reference · No measured ablation report available</p></div></div></section>
}
