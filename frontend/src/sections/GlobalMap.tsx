import { useBackend } from '../hooks/useBackend'
import { SectionHead } from '../components/ui'
import { API_BASE } from '../lib/api'
export function GlobalMap() {
  const {status,inferenceReady} = useBackend()
  return <section id="global-map" className="section research-detail"><div className="container"><SectionHead kicker="YOUR LOCAL WORKSPACE" title={<>A focused system.<br/><em>A clear connection.</em></>} lede="Camera capture and landmarks run in the browser. Available alphabet inference runs on your configured backend."/><div className="research-schematic"><div className="research-topology"><div><small>IN YOUR BROWSER</small><h3>Capture & landmarks</h3><p>Camera access starts with your action.</p></div><span aria-hidden="true">↔</span><div><small>CONFIGURED API</small><h3>{status==='online'?'Backend connected':status==='checking'||status==='idle'?'Checking connection':'Backend offline'}</h3><p>{inferenceReady?'Alphabet inference available':'Alphabet readiness unconfirmed'}</p></div></div><p className="mono-note">API: {API_BASE || 'Same origin'} · Camera recordings are not saved by the frontend</p></div></div></section>
}
