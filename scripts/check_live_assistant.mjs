// Browser software checks: explicit API fixtures, genuine public video replay,
// real MediaPipe tracking. No fixture result is recognition/accuracy evidence.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, writeFile } from 'node:fs/promises'
import path from 'node:path'
const base = process.env.LIPNET_QA_BASE ?? 'http://127.0.0.1:8001'
const chromePath = process.env.LIPNET_CHROME_PATH ?? [process.env.PROGRAMFILES, process.env['PROGRAMFILES(X86)'], process.env.LOCALAPPDATA]
  .filter(Boolean).map(root => path.join(root, 'Google/Chrome/Application/chrome.exe')).find(existsSync)
if (!chromePath) throw new Error('Chrome not found. Set LIPNET_CHROME_PATH to the installed executable.')
const profile = await mkdtemp(path.resolve('.qa/live-assistant-chrome-'))
const chrome = spawn(chromePath, [
  '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
  '--disable-background-timer-throttling','--disable-renderer-backgrounding','--disable-backgrounding-occluded-windows',
  '--remote-debugging-port=0','--user-data-dir='+profile,
  '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',
  '--use-file-for-fake-video-capture='+path.resolve('.qa/afternoon-paced-replay.y4m'),'about:blank',
], { windowsHide:true, stdio:['ignore','ignore','pipe'] })
class CDP {
  constructor(socket) {
    this.socket=socket;this.next=0;this.pending=new Map();this.errors=[]
    socket.addEventListener('message',event=>{
      const d=JSON.parse(event.data)
      if(d.id){const p=this.pending.get(d.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result)}
      else if(d.method==='Runtime.exceptionThrown')this.errors.push(d.params)
    })
  }
  call(method,params={}) { const id=++this.next;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>{this.pending.delete(id);reject(new Error('CDP timeout: '+method))},30000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params}))}) }
  async eval(expression) { const r=await this.call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value }
}
const sleep = ms => new Promise(r=>setTimeout(r,ms))
async function connect(url) {const s=new WebSocket(url);await new Promise((resolve,reject)=>{s.addEventListener('open',resolve,{once:true});s.addEventListener('error',reject,{once:true})});return new CDP(s)}
const checks=[], captures=[], observations=[]
let browser,page
const check=(value,name)=>{if(!value)throw new Error(name);checks.push(name);console.log('PASS '+name)}
const fixture = `
window.__qa={requests:[],streams:[],cameraCalls:0,cameraMode:'normal',exports:new Map(),serial:0,starts:[]};
const q=window.__qa, originalFetch=window.fetch.bind(window), originalCamera=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
navigator.mediaDevices.getUserMedia=async(...args)=>{
  q.cameraCalls++;const t=performance.now();
  if(q.cameraMode==='denied')throw new DOMException('fixture denied','NotAllowedError');
  if(q.cameraMode==='missing')throw new DOMException('fixture missing','NotFoundError');
  if(q.cameraMode==='busy')throw new DOMException('fixture busy','NotReadableError');
  const s=await originalCamera(...args);q.streams.push(s);q.starts.push(performance.now()-t);
  if(q.cameraMode==='delayed')await new Promise(r=>q.releasePermission=r);
  return s;
};
const exportCanvas=HTMLCanvasElement.prototype.toDataURL;
HTMLCanvasElement.prototype.toDataURL=function(...args){const image=exportCanvas.apply(this,args);q.exports.set(image,++q.serial);return image};
window.fetch=(url,init)=>{
  const u=new URL(url,location.href), p=u.pathname;
  const json=value=>Promise.resolve(new Response(JSON.stringify(value),{status:200,headers:{'Content-Type':'application/json'}}));
  if(p==='/health'||p==='/api/health')return json({status:'ok',model_loaded:true,inference_ready:true,model_kind:'Software fixture alphabet'});
  if(p==='/api/word/health')return json({model_loaded:true,inference_ready:true,research_activation_gate_passed:true,model_kind:'Software fixture eight words',vocabulary:['choose','connection','hello','navigation','next','previous','start','stop']});
  if(p==='/api/word/lrw500/health')return json({model_loaded:true,inference_ready:true,state:'ready',model_kind:'Software fixture 500 words',vocabulary:Array.from({length:500},(_,i)=>'WORD'+i)});
  if(p==='/api/dataset/stats')return json({videos:0});
  if(p.startsWith('/api/predict/word')){
    const body=JSON.parse(init.body), ids=body.frames.map(f=>q.exports.get(f));
    const entry={path:p,frames:body.frames.length,ids,metadata:{...body,frames:undefined},sentAt:performance.now(),signal:init.signal};q.requests.push(entry);
    return new Promise(resolve=>entry.respond=(value,status=200)=>{entry.respondedAt=performance.now();resolve(new Response(JSON.stringify(value),{status,headers:{'Content-Type':'application/json'}}))});
  }
  return originalFetch(url,init);
};`
try {
  const address=await new Promise((resolve,reject)=>{let log='';const timer=setTimeout(()=>reject(new Error('Chrome startup timeout')),30000);chrome.on('error',reject);chrome.stderr.on('data',b=>{log+=b;const m=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1])}})})
  browser=await connect(address)
  const target=await(await fetch('http://'+new URL(address).host+'/json/new?about:blank',{method:'PUT'})).json()
  page=await connect(target.webSocketDebuggerUrl)
  await page.call('Page.enable');await page.call('Runtime.enable')
  await page.call('Page.addScriptToEvaluateOnNewDocument',{source:fixture})
  await page.call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false})
  await page.call('Page.navigate',{url:base+'/demo/live'})
  const wait=async(expression,ms=15000)=>{const end=Date.now()+ms;while(Date.now()<end){if(await page.eval(expression))return true;await sleep(80)}return false}
  const click=async label=>page.eval(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b)throw new Error('Missing button '+${JSON.stringify(label)});if(b.disabled)throw new Error('Disabled button '+${JSON.stringify(label)});b.click()})()`)
  const body=()=>page.eval('document.body.innerText')
  const phase=state=>`!!document.querySelector('.ws-assistant-capture.state-${state.toLowerCase()}')`
  const readyFace=async()=>check(await wait(`!![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Start Word'&&!b.disabled)`,60000),'real MediaPipe yields a usable face and mouth')
  const response={task:'isolated_word',predicted_word:'hello',display_word:'hello',confidence:.87,uncertain:false,top_predictions:[{word:'hello',prob:.87}],latency_ms:123.4}
  const reply=extra=>page.eval(`window.__qa.requests.at(-1).respond(${JSON.stringify({...response,...extra})})`)
  const capture=async(action='Start Word',lrw=false)=>{
    await click(action)
    check(await wait(phase('CAPTURING')),'explicit '+action+' begins capture')
    check(await wait(`document.querySelector('.ws-assistant-capture progress')?.value>=${lrw?14:9}`,7000),'capture collects sufficient fresh frames')
    await click('Finish Word')
    check(await wait(phase('PROCESSING')),'Finish Word enters processing')
    check(await wait('window.__qa.requests.length>0 && !window.__qa.requests.at(-1).respondedAt'),'one pending word request')
    captures.push(await page.eval('(()=>{const r=window.__qa.requests.at(-1);return {path:r.path,frames:r.frames,ids:r.ids,metadata:r.metadata}})()'))
  }
  check(await wait(`document.querySelector('.ws-live-modes>button[aria-pressed=true]')?.textContent==='8 words'`),'eight-word model remains default and selectable')
  check(await page.eval('window.__qa.cameraCalls===0 && document.querySelector("video").srcObject===null'),'camera stays off until requested')
  for(const [mode,message] of [['denied','Camera permission was denied'],['missing','No camera was found'],['busy','camera is busy']]) {
    await page.eval(`window.__qa.cameraMode=${JSON.stringify(mode)}`)
    await click(mode==='denied'?'Start camera':'Retry camera')
    check(await wait(`document.body.innerText.includes(${JSON.stringify(message)})`),'friendly '+mode+' camera failure')
  }
  await page.eval(`window.__qa.cameraMode='normal'`)
  await page.eval(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Retry camera');b.click();b.click()})()`)
  await readyFace()
  check(await page.eval('window.__qa.streams.length===1'),'duplicate Start creates one camera stream')
  check(await page.eval('window.__qa.requests.length===0'),'camera preview alone sends no predictions')
  await capture()
  check(await page.eval(`document.body.innerText.includes('Analyzing lip movements') && [...document.querySelectorAll('.ws-live-modes>button')].every(b=>b.disabled)`),'processing indicator and conflicting mode controls disabled')
  await sleep(300)
  check(await page.eval('window.__qa.requests.length===1'),'no duplicate requests while processing')
  await reply({})
  check(await wait(phase('WAITING_NEXT')),'result transitions to waiting for Next Word')
  check(await page.eval(`document.querySelector('.ws-assistant-word').textContent==='hello' && document.querySelectorAll('.ws-assistant-history>li').length===1`),'prediction and session history rendered')
  await sleep(400)
  check(await page.eval('window.__qa.requests.length===1'),'waiting never automatically captures or predicts')
  await capture('Next Word')
  check(Math.min(...captures[1].ids)>Math.max(...captures[0].ids),'second request contains newly exported frames only')
  await reply({uncertain:true,confidence:.31})
  check(await wait(phase('WAITING_NEXT')),'uncertain response completes without sticking')
  check(await page.eval(`document.querySelector('.ws-assistant-word').textContent==='Uncertain' && document.querySelectorAll('.ws-assistant-history>li').length===2`),'rejected candidate stays uncertain and history is preserved')
  await page.eval(`document.querySelector('.ws-assistant-feedback button').click()`)
  check(await page.eval(`document.querySelector('.ws-assistant-feedback button').getAttribute('aria-pressed')==='true'`),'explicit session feedback can be recorded')
  await capture('Retry Word')
  await reply({confidence:null,latency_ms:undefined})
  check(await wait(phase('WAITING_NEXT')),'retry completes as a separate attempt')
  check(await page.eval(`document.querySelector('.ws-assistant').innerText.includes('Confidence unavailable') && document.querySelectorAll('.ws-assistant-history>li').length===3 && window.__qa.streams.length===1`),'retry retains camera and history; absent confidence is unavailable')
  const {data}=await page.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true})
  await writeFile('.qa/live-assistant-desktop.png',Buffer.from(data,'base64'))
  await click('Stop camera')
  check(await wait('window.__qa.streams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))'),'Stop releases every camera track')
  check(await page.eval(`document.querySelector('.ws-assistant-word').textContent==='hello' && document.querySelectorAll('.ws-assistant-history>li').length===3`),'Stop retains last prediction and session history')
  await click('Start camera');await readyFace();await capture()
  await click('Stop camera');await reply({predicted_word:'stale'})
  await sleep(150)
  check(await page.eval(`document.querySelectorAll('.ws-assistant-history>li').length===3 && !document.body.innerText.includes('stale') && window.__qa.requests.at(-1).signal.aborted`),'Stop aborts processing and discards late response')
  await click('Start camera');await readyFace();await capture()
  await page.eval(`window.__qa.requests.at(-1).respond({detail:'Word analysis failed. Retry the word.'},500)`)
  check(await wait(phase('ERROR')),'inference exception produces recoverable ERROR')
  await capture('Retry Word');await page.eval(`window.__qa.requests.at(-1).respond({unexpected:true})`)
  check(await wait(phase('ERROR')),'invalid backend response leaves PROCESSING')
  await click('Back to Ready')
  await click('Start Word')
  await page.eval(`window.__qa.streams.at(-1).getVideoTracks()[0].dispatchEvent(new Event('ended'))`)
  check(await wait(`document.body.innerText.includes('camera disconnected')`),'device disconnection is handled during capture')
  check(await page.eval('window.__qa.streams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))'),'disconnect releases all tracks')
  await click('Reset session')
  check(await page.eval(`document.querySelectorAll('.ws-assistant-history>li').length===0 && [...document.querySelectorAll('.ws-assistant-stats dd')].slice(0,6).every(e=>e.textContent==='0')`),'explicit reset clears history and totals')
  await page.eval(`window.__qa.cameraMode='delayed'`)
  await click('Start camera')
  check(await wait('!!window.__qa.releasePermission'),'pending camera permission can be cancelled')
  await click('Cancel')
  await page.eval(`window.__qa.cameraMode='normal';window.__qa.cancelledStream=window.__qa.streams.at(-1)`)
  await click('Start camera')
  await page.eval(`window.__qa.releasePermission()`)
  await readyFace()
  check(await page.eval('window.__qa.cancelledStream.getTracks().every(t=>t.readyState==="ended")'),'late cancelled permission releases tracks without blocking a new start')
  await click('Stop camera')
  // Both word models continue to use their own contracts.
  await click('500 words');await click('Start camera');await readyFace();await capture('Start Word',true)
  const last=captures.at(-1)
  check(last.path.endsWith('/lrw500') && last.metadata.final===true && last.metadata.landmarks.length===last.frames && last.metadata.timestamps_ms.length===last.frames,'500-word request preserves full-face/timestamp/final contract')
  await reply({predicted_word:'WORD1',diagnostics:{temporal_consensus:{confirmed:true,views:[14,12]}}})
  check(await wait(phase('WAITING_NEXT')),'500-word response waits for explicit Next Word')
  await click('Next Word')
  await page.eval(`document.querySelector('a[href="/demo/dataset"]').click()`)
  check(await wait('!document.querySelector(".ws-live")'),'navigation leaves the live screen')
  check(await page.eval('window.__qa.streams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))'),'navigation/unmount releases camera tracks')
  observations.push(await page.eval('({camera_start_ms:window.__qa.starts,requests:window.__qa.requests.map(r=>({path:r.path,frames:r.frames,elapsed_ms:r.respondedAt?r.respondedAt-r.sentAt:null}))})'))
  await page.call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true})
  for(const route of ['/demo/live','/demo/overview','/demo/dataset','/demo/training','/demo/testing','/demo/models','/demo/experiments','/demo/pipeline','/demo/docs','/science','/research','/']) {
    await page.call('Page.navigate',{url:base+route});check(await wait('document.querySelector("h1")!==null',25000),'route renders '+route)
    check(await page.eval('document.documentElement.scrollWidth<=innerWidth+1'),'no mobile overflow '+route)
    if(route==='/demo/live') {const {data}=await page.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await writeFile('.qa/live-assistant-mobile.png',Buffer.from(data,'base64'))}
  }
  check(page.errors.length===0,'no uncaught browser exceptions')
  await writeFile('.qa/live-assistant-browser.json',JSON.stringify({scope:'Software fixtures for API; real MediaPipe on genuine public AFTERNOON camera replay. Not physical-webcam accuracy or model timing.',checks_passed:checks.length,checks,captures,observations,uncaught_errors:page.errors},null,2))
  console.log(`${checks.length} browser checks passed`)
} catch(error) {
  await writeFile('.qa/live-assistant-browser-failure.json',JSON.stringify({error:String(error),checks,captures,exceptions:page?.errors,telemetry:page?await page.eval('document.body.innerText').catch(()=>null):null},null,2))
  throw error
} finally {
  if(page)page.socket.close()
  if(browser){try{await browser.call('Browser.close')}catch{}browser.socket.close()}
  if(chrome.exitCode===null)chrome.kill()
}
