import { useState } from 'react'
import { CONFORMER } from '../data/architecture'
import { SectionHead } from '../components/ui'
const units = [
  {name:'Feed forward',description:'A position-wise network transforms features within each time step.'},
  {name:'Self attention',description:'Attention relates visual observations across the temporal sequence.'},
  {name:'Convolution',description:'A convolution module captures local patterns in the sequence.'},
]
export function ConformerExplorer() {
  const [block,setBlock] = useState(0)
  const [unit,setUnit] = useState(0)
  return <section id="conformer-explorer" className="section research-detail"><div className="container"><SectionHead kicker="ARCHITECTURE EXPLORER" title={<>The context between <em>the frames.</em></>} lede="Inspect the Conformer research architecture. The connected experimental alphabet classifier uses a separate CNN-BiLSTM architecture."/><div className="research-schematic"><div className="research-blocks" aria-label="Research encoder blocks">{Array.from({length:CONFORMER.blocks},(_,index)=><button className={block===index?'is-active':''} key={index} aria-pressed={block===index} onClick={()=>setBlock(index)}><small>ENCODER</small><strong>{String(index+1).padStart(2,'0')}</strong><span>Conformer</span></button>)}</div><div className="research-unit-tabs">{units.map((item,index)=><button key={item.name} onClick={()=>setUnit(index)} aria-pressed={unit===index} className={unit===index?'is-active':''}>{item.name}</button>)}</div><h3>Block {block+1} / {units[unit].name}</h3><p className="research-body">{units[unit].description}</p><p className="mono-note">Reference configuration · {CONFORMER.blocks} blocks · d={CONFORMER.dModel} · {CONFORMER.heads} attention heads</p></div></div></section>
}
