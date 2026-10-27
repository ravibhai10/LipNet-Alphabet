import { spawn } from 'node:child_process'
import { mkdir, mkdtemp, writeFile } from 'node:fs/promises'
import { existsSync } from 'node:fs'
import path from 'node:path'
const output = path.resolve('.qa')
await mkdir(output, { recursive: true })
const profile = await mkdtemp(path.join(output, 'chrome-profile-'))
const chromePath = [
  process.env.CHROME_PATH,
  path.join(process.env.PROGRAMFILES || '', 'Google/Chrome/Application/chrome.exe'),
  path.join(process.env['PROGRAMFILES(X86)'] || '', 'Microsoft/Edge/Application/msedge.exe'),
].find(candidate => candidate && existsSync(candidate))
if (!chromePath) throw new Error('Set CHROME_PATH to an installed Chromium browser to run local browser QA.')
const chrome = spawn(chromePath, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check', '--remote-debugging-port=0', '--user-data-dir=' + profile, 'about:blank'], { windowsHide: true, stdio: ['ignore', 'ignore', 'pipe'] })
class CDP {
  constructor(socket) {
    this.socket = socket; this.next = 0; this.pending = new Map(); this.events = []
    socket.addEventListener('message', event => {
      const data = JSON.parse(event.data)
      if (data.id) {
        const item = this.pending.get(data.id)
        if (!item) return
        clearTimeout(item.timer); this.pending.delete(data.id)
        data.error ? item.reject(new Error(JSON.stringify(data.error))) : item.resolve(data.result)
      } else this.events.push(data)
    })
  }
  call(method, params = {}) {
    const id = ++this.next
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { this.pending.delete(id); reject(new Error('CDP timeout: ' + method)) }, 30000)
      this.pending.set(id, { resolve, reject, timer }); this.socket.send(JSON.stringify({ id, method, params }))
    })
  }
  async eval(expression) {
    const data = await this.call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
    if (data.exceptionDetails) throw new Error(data.exceptionDetails.text + ': ' + data.exceptionDetails.exception?.description)
    return data.result.value
  }
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms))
let browser, page
const checks = []
const assert = (name, condition, detail) => { checks.push({ name, passed: !!condition, detail }); if (!condition) throw new Error(name + ': ' + JSON.stringify(detail)) }
try {
  const browserURL = await new Promise((resolve, reject) => {
    let log = ''
    const timer = setTimeout(() => reject(new Error('Chrome did not start: ' + log.slice(-1000))), 25000)
    chrome.on('error', reject)
    chrome.stderr.on('data', data => { log += data; const found = log.match(/DevTools listening on (ws:\/\/[^\s]+)/); if (found) { clearTimeout(timer); resolve(found[1]) } })
    chrome.on('exit', code => { if (code) reject(new Error('Chrome exited ' + code + ': ' + log.slice(-1500))) })
  })
  const browserSocket = new WebSocket(browserURL)
  await new Promise((resolve, reject) => { browserSocket.addEventListener('open', resolve, {once:true}); browserSocket.addEventListener('error', reject, {once:true}) })
  browser = new CDP(browserSocket)
  const address = new URL(browserURL).host
  const target = await (await fetch('http://' + address + '/json/new?about:blank', { method: 'PUT' })).json()
  const socket = new WebSocket(target.webSocketDebuggerUrl)
  await new Promise((resolve, reject) => { socket.addEventListener('open', resolve, {once:true}); socket.addEventListener('error', reject, {once:true}) })
  page = new CDP(socket)
  await page.call('Page.enable'); await page.call('Runtime.enable'); await page.call('Network.enable')
  await page.call('Page.addScriptToEvaluateOnNewDocument', { source: "window.__cameraCalls=0; if(navigator.mediaDevices){const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices); navigator.mediaDevices.getUserMedia=(...args)=>{window.__cameraCalls++; return original(...args)}};" })
  const routes = ['overview', 'live', 'dataset', 'training', 'testing', 'models', 'experiments', 'pipeline', 'docs']
  async function navigate(route) {
    await page.call('Page.navigate', { url: 'http://127.0.0.1:5173/demo/' + route })
    for (let i=0;i<80;i++) {
      const ready = await page.eval("location.pathname.endsWith('/" + route + "') && !!document.querySelector('.ws-page h1')")
      if (ready) break
      await sleep(100)
    }
    await sleep(450)
    await page.eval('document.fonts.ready.then(()=>true)')
  }
  async function screenshot(name) { const {data} = await page.call('Page.captureScreenshot', { format:'png', captureBeyondViewport:false }); await writeFile(path.join(output,name), Buffer.from(data,'base64')) }
  if (!process.env.MATH_ONLY) {
  for (const viewport of [{width:1366,height:768,mobile:false},{width:900,height:1024,mobile:false},{width:390,height:844,mobile:true}]) {
    await page.call('Emulation.setDeviceMetricsOverride', {...viewport,deviceScaleFactor:1})
    for (const route of routes) {
      await navigate(route)
      const state = await page.eval("({title:document.querySelector('h1')?.innerText,error:document.body.innerText.includes('Unexpected Application Error'),cameraCalls:window.__cameraCalls,overflow:document.querySelector('.ws-content').scrollWidth>document.querySelector('.ws-content').clientWidth+1,palette:getComputedStyle(document.querySelector('.workspace')).getPropertyValue('--ws-accent').trim(),backend:document.querySelector('.ws-connection')?.innerText})")
      assert(viewport.width + '/' + route + ' renders without error, overflow, or camera prompt', !!state.title && !state.error && !state.overflow && state.cameraCalls===0 && state.palette==='#8fdfcb', state)
      if (viewport.width !== 900 && ['live','overview','experiments'].includes(route)) await screenshot((viewport.mobile?'mobile-':'desktop-') + route + '.png')
    }
    if (viewport.mobile) {
      await page.eval("document.getElementById('ws-menu-button').click()")
      assert('Mobile navigation opens', await page.eval("document.querySelector('.ws-sidebar').classList.contains('mobile-open')"))
      await page.call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27})
      assert('Escape closes mobile navigation', !await page.eval("document.querySelector('.ws-sidebar').classList.contains('mobile-open')"))
    }
  }
  await page.call('Emulation.setDeviceMetricsOverride',{width:1366,height:768,deviceScaleFactor:1,mobile:false})
  await navigate('live')
  await page.eval("navigator.mediaDevices.getUserMedia=async()=>{window.__cameraCalls++;throw new DOMException('QA denial','NotAllowedError')}; [...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Start camera').click()")
  await sleep(400)
  assert('Camera permission denial is actionable', await page.eval("document.body.innerText.includes('site settings') && document.body.innerText.includes('Retry camera') && !document.body.innerText.includes('Camera streaming')"))
  await screenshot('desktop-camera-denied.png')
  await navigate('live')
  // A synthetic canvas stream checks lifecycle without using a physical camera.
  await page.eval("window.__qaCanvas=document.createElement('canvas');__qaCanvas.width=640;__qaCanvas.height=360;window.__qaTimer=setInterval(()=>{const ctx=__qaCanvas.getContext('2d');ctx.fillStyle='#172220';ctx.fillRect(0,0,640,360)},100);window.__qaStream=__qaCanvas.captureStream(10);navigator.mediaDevices.getUserMedia=async()=>{window.__cameraCalls++;return window.__qaStream};[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Start camera').click()")
  for(let i=0;i<150;i++){if(await page.eval("document.body.innerText.includes('Camera streaming') && !document.body.innerText.includes('LOADING LANDMARKS')"))break;await sleep(100)}
  const lifecycle = await page.eval("({calls:window.__cameraCalls,active:window.__qaStream.active,streaming:document.body.innerText.includes('Camera streaming'),runtimeError:document.body.innerText.includes('Landmark pipeline needs attention'),noPredictedLetter:document.querySelector('.ws-live-letter').textContent.trim()==='—'})")
  assert('Synthetic camera stream starts without invented prediction', lifecycle.calls===1 && lifecycle.active && lifecycle.streaming && !lifecycle.runtimeError && lifecycle.noPredictedLetter,lifecycle)
  await page.eval("[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Stop camera').click();clearInterval(window.__qaTimer)")
  assert('Stop releases camera tracks', await page.eval("!window.__qaStream.active && !document.querySelector('video').srcObject && document.body.innerText.includes('Camera off')"))
  await page.eval("navigator.mediaDevices.getUserMedia=async()=>{window.__qaStream=window.__qaCanvas.captureStream(10);return window.__qaStream};[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Start camera').click()")
  await sleep(300)
  await page.eval("window.__qaStream.getVideoTracks()[0].dispatchEvent(new Event('ended'))")
  await sleep(200)
  assert('Camera disconnect during startup clears state and releases its stream',await page.eval("!window.__qaStream.active && document.body.innerText.includes('Camera unavailable') && document.querySelector('.ws-live-letter').textContent.trim()==='—'"))
  await page.eval("navigator.mediaDevices.getUserMedia=async()=>{window.__qaStream=window.__qaCanvas.captureStream(10);return window.__qaStream};[...document.querySelectorAll('button')].find(b=>['Start camera','Retry camera'].includes(b.textContent.trim())).click()")
  await sleep(300)
  await page.eval("document.querySelector('a[href=\"/demo/overview\"]').click()")
  await sleep(400)
  assert('Route unmount releases camera tracks',await page.eval("!window.__qaStream.active && location.pathname==='/demo/overview'"))
  await navigate('live')
  await page.eval("window.__pendingResolve=null;navigator.mediaDevices.getUserMedia=()=>new Promise(resolve=>{window.__pendingResolve=resolve});[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Start camera').click()")
  await sleep(150)
  await page.eval("[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Cancel').click();const canvas=document.createElement('canvas');window.__lateStream=canvas.captureStream();window.__pendingResolve(window.__lateStream)")
  await sleep(250)
  assert('Cancelled permission request cannot revive a late stream', await page.eval("!window.__lateStream.active && !document.querySelector('video').srcObject && document.body.innerText.includes('Camera off')"))
  await navigate('live')
  await page.eval("window.__realFetch=window.fetch;window.fetch=async(...args)=>{if(String(args[0]).includes(':8000'))throw new TypeError('QA offline');return __realFetch(...args)};document.querySelector('[aria-label=\"Refresh backend connection\"]').click()")
  for (let i=0;i<40;i++) {
    if (await page.eval("document.body.innerText.includes('Backend offline')")) break
    await sleep(150)
    await page.eval("document.querySelector('[aria-label=\"Refresh backend connection\"]').click()")
  }
  assert('Offline backend is separate from camera state',await page.eval("document.body.innerText.includes('Backend offline') && document.body.innerText.includes('Camera off') && document.querySelector('.ws-live-letter').textContent.trim()==='—'"))
  await screenshot('desktop-backend-offline.png')
  await navigate('live')
  const apiChecks = await page.eval("(async()=>{const api=await import('/src/lib/api.ts');const saved=window.fetch;const results=[];try{for(const code of [503,404]){window.fetch=async()=>new Response(JSON.stringify({detail:'QA HTTP '+code}),{status:code,headers:{'Content-Type':'application/json'}});try{await api.requestPrediction(['fixture']);results.push(false)}catch(error){results.push(error.status===code && error.message==='QA HTTP '+code)}}window.fetch=async()=>new Response('invalid json',{status:200});try{await api.requestPrediction(['fixture']);results.push(false)}catch{results.push(true)}window.fetch=(_url,init)=>new Promise((_resolve,reject)=>init.signal.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError'))));try{await api.fetchWithTimeout('/fixture',{},15);results.push(false)}catch(error){results.push(error.name==='AbortError')}return results}finally{window.fetch=saved}})()")
  assert('API client rejects HTTP 503, HTTP 404, invalid JSON, and timed out requests',apiChecks.every(Boolean),apiChecks)
  await page.call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]})
  assert('Reduced motion disables page animation',await page.eval("getComputedStyle(document.querySelector('.ws-page')).animationName==='none'"))
  await page.call('Emulation.setEmulatedMedia',{features:[]})
  for(const viewport of [{width:1366,height:768,mobile:false},{width:390,height:844,mobile:true}]){
    await page.call('Emulation.setDeviceMetricsOverride',{...viewport,deviceScaleFactor:1})
    await page.call('Page.navigate',{url:'http://127.0.0.1:5173/'})
    for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('.hero-title')"))break;await sleep(100)}
    await sleep(1800)
    const landing=await page.eval("({hero:!!document.querySelector('.hero-title'),cameraCalls:window.__cameraCalls,accent:getComputedStyle(document.documentElement).getPropertyValue('--accent').trim(),ink:getComputedStyle(document.documentElement).getPropertyValue('--ink').trim(),fakeEvents:document.body.innerText.includes('Each dot is a recognition event'),fakeAccuracy:document.body.innerText.includes('ESTIMATED OVERALL ACCURACY')})")
    assert('Landing page '+viewport.width+' retains palette and honest states',landing.hero && landing.cameraCalls===0 && landing.accent==='#8fdfcb' && landing.ink==='#edebe4' && !landing.fakeEvents && !landing.fakeAccuracy,landing)
    await screenshot(viewport.mobile?'mobile-landing.png':'desktop-landing.png')
    const story = await page.eval("({scenes:document.querySelectorAll('[data-story-scene]').length,sidebar:!!document.querySelector('.ws-sidebar'),detailsOpen:document.querySelector('.story-technical').open,overflow:document.documentElement.scrollWidth>innerWidth+1,missingAnchors:[...document.querySelectorAll('.story-experience a[href^=\"#\"]')].map(a=>a.getAttribute('href').slice(1)).filter(id=>id&&!document.getElementById(id)),noPrediction:document.querySelector('.story-prediction').textContent.trim()==='—',headlines:[...document.querySelectorAll('[data-story-scene] h1,[data-story-scene] h2')].map(e=>e.innerText)})")
    assert('Story '+viewport.width+' has six scenes, valid anchors, hidden diagnostics, and no invented output',story.scenes===6&&!story.sidebar&&!story.detailsOpen&&!story.overflow&&!story.missingAnchors.length&&story.noPrediction,story)
    const directLinks = await page.eval("['/science','/demo/live'].every(path=>{const a=document.querySelector('.story-nav > nav a[href=\"'+path+'\"]');return a&&a.getClientRects().length>0})")
    assert('Story '+viewport.width+' exposes Science and Live Demo directly in the header',directLinks)
    await page.eval("document.querySelector('.story-pages-toggle').click()")
    await sleep(300)
    const directory = await page.eval("({open:!document.getElementById('story-page-directory').hidden,expanded:document.querySelector('.story-pages-toggle').getAttribute('aria-expanded'),paths:[...document.querySelectorAll('.story-directory-group a')].map(a=>a.getAttribute('href')),overflow:document.documentElement.scrollWidth>innerWidth+1})")
    const expectedPaths=['/','/science','/research',...routes.map(route=>'/demo/'+route)]
    assert('All pages menu '+viewport.width+' links every existing project page without overflow',directory.open&&directory.expanded==='true'&&!directory.overflow&&expectedPaths.every(path=>directory.paths.includes(path)),directory)
    assert('All pages menu '+viewport.width+' is opaque and receives pointer input',await page.eval("(()=>{const menu=document.getElementById('story-page-directory'),a=menu.querySelector('a[href=\"/science\"]'),r=a.getBoundingClientRect();return getComputedStyle(menu).opacity==='1'&&document.elementFromPoint(r.x+r.width/2,r.y+r.height/2)?.closest('a')===a})()"))
    await screenshot(viewport.mobile?'mobile-all-pages.png':'desktop-all-pages.png')
    await page.call('Input.dispatchKeyEvent',{type:'keyDown',key:'Escape',code:'Escape',windowsVirtualKeyCode:27})
    assert('All pages menu '+viewport.width+' closes with Escape and restores focus',await page.eval("document.getElementById('story-page-directory').hidden&&document.activeElement===document.querySelector('.story-pages-toggle')"))
    const frame=await page.eval("document.querySelector('.story-hero-face .hero-visual canvas').toDataURL()")
    await sleep(180)
    assert('Original face particles animate at '+viewport.width,await page.eval("document.querySelector('.story-hero-face .hero-visual').dataset.motion==='running'&&document.querySelector('.story-hero-face .hero-visual canvas').toDataURL()!=="+JSON.stringify(frame)))
    await page.eval("document.querySelector('.story-motion-toggle').click();window.__pausedCanvas=document.querySelector('.story-hero-face .hero-visual canvas').toDataURL()")
    await sleep(180)
    assert('Face animation can be paused at '+viewport.width,await page.eval("document.querySelector('.story-hero-face .hero-visual').dataset.motion==='paused'&&document.querySelector('.story-hero-face .hero-visual canvas').toDataURL()===window.__pausedCanvas"))
    await page.eval("document.querySelector('.story-motion-toggle').click()")
    for(const scene of ['idea','method','demo','evidence','journey']) {
      await page.eval("document.getElementById('"+scene+"').scrollIntoView({behavior:'instant',block:'start'})")
      await sleep(700)
      await screenshot((viewport.mobile?'mobile-':'desktop-')+'story-'+scene+'.png')
    }
  }
  await page.call('Emulation.setDeviceMetricsOverride',{width:1366,height:900,deviceScaleFactor:1,mobile:false})
  await page.eval("document.querySelector('.story-nav a[href=\"/science\"]').click()")
  for(let i=0;i<100;i++){if(await page.eval("location.pathname==='/science'&&!!document.querySelector('.science-nav')"))break;await sleep(100)}
  assert('Header Science link opens the actual Science Lab at the top',await page.eval("location.pathname==='/science'&&!!document.querySelector('.science-container h1')&&window.scrollY<5"))
  await page.eval("document.querySelector('.science-nav a[href=\"/\"]').click()")
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('.story-nav')"))break;await sleep(100)}
  await page.eval("document.querySelector('.story-nav a[href=\"/demo/live\"]').click()")
  for(let i=0;i<100;i++){if(await page.eval("location.pathname==='/demo/live'&&!!document.querySelector('.ws-live-letter')"))break;await sleep(100)}
  assert('Header Live Demo link opens the full studio with Science linked in its sidebar',await page.eval("location.pathname==='/demo/live'&&!!document.querySelector('.ws-sidebar a[href=\"/science\"]')&&!!document.querySelector('.ws-sidebar a[href=\"/research\"]')"))
  await page.call('Page.navigate',{url:'http://127.0.0.1:5173/'})
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('.story-demo')"))break;await sleep(100)}
  await page.eval("document.getElementById('demo').scrollIntoView({behavior:'instant'});navigator.mediaDevices.getUserMedia=async()=>{window.__cameraCalls++;throw new DOMException('QA denial','NotAllowedError')};[...document.querySelectorAll('.story-demo button')].find(b=>b.textContent.trim()==='Start camera').click()")
  await sleep(350)
  assert('Story camera denial stays actionable without a prediction',await page.eval("window.__cameraCalls===1 && document.querySelector('.story-error')?.textContent.includes('site settings') && document.querySelector('.story-prediction').textContent.trim()==='—'"))
  await page.eval("window.__storyCanvas=document.createElement('canvas');__storyCanvas.width=640;__storyCanvas.height=360;window.__storyTimer=setInterval(()=>{const c=__storyCanvas.getContext('2d');c.fillStyle='#172220';c.fillRect(0,0,640,360)},100);window.__storyStream=__storyCanvas.captureStream(10);navigator.mediaDevices.getUserMedia=async()=>{window.__cameraCalls++;return __storyStream};[...document.querySelectorAll('.story-demo button')].find(b=>b.textContent.trim()==='Retry camera').click()")
  for(let i=0;i<100;i++){if(await page.eval("document.querySelector('.story-camera-caption').textContent.includes('Camera on')"))break;await sleep(100)}
  assert('Story camera preview uses a real stream and does not invent a letter',await page.eval("__storyStream.active && document.querySelector('.story-prediction').textContent.trim()==='—' && document.querySelector('.story-camera video').srcObject===__storyStream"))
  await page.eval("[...document.querySelectorAll('.story-demo button')].find(b=>b.textContent.trim()==='Stop camera').click();clearInterval(__storyTimer)")
  assert('Story stop releases the device and clears output',await page.eval("!__storyStream.active && !document.querySelector('.story-camera video').srcObject && document.querySelector('.story-prediction').textContent.trim()==='—'"))
  await page.eval("window.__storyStream=__storyCanvas.captureStream(10);navigator.mediaDevices.getUserMedia=async()=>__storyStream;[...document.querySelectorAll('.story-demo button')].find(b=>b.textContent.trim()==='Start camera').click()")
  await sleep(250)
  await page.eval("document.querySelector('a[href=\"/demo/overview\"]').click()")
  await sleep(300)
  assert('Story-to-workspace navigation releases the camera',await page.eval("location.pathname==='/demo/overview' && !__storyStream.active"))
  await page.call('Page.navigate',{url:'http://127.0.0.1:5173/research'})
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('.hero-title')"))break;await sleep(100)}
  assert('Preserved research exhibits render without a camera prompt',await page.eval("!!document.querySelector('.hero-title') && window.__cameraCalls===0 && !document.body.innerText.includes('Unexpected Application Error')"))
  await page.call('Page.navigate',{url:'http://127.0.0.1:5173/'})
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('.story-demo')"))break;await sleep(100)}
  await page.eval("window.__realFetch=window.fetch;window.fetch=async(...args)=>{if(String(args[0]).includes(':8000'))throw new TypeError('QA offline');return __realFetch(...args)}")
  for(let i=0;i<40;i++){
    await page.eval("document.querySelector('[aria-label=\"Refresh story backend connection\"]').click()")
    if(await page.eval("document.querySelector('.story-connection').textContent.includes('Backend offline')"))break
    await sleep(150)
  }
  assert('Story remains usable offline without invented predictions',await page.eval("document.querySelector('.story-connection').textContent.includes('Backend offline') && document.querySelector('.story-prediction').textContent.trim()==='—' && !![...document.querySelectorAll('.story-demo button')].find(b=>b.textContent.trim()==='Start camera')"))
  await page.call('Emulation.setEmulatedMedia',{features:[{name:'prefers-reduced-motion',value:'reduce'}]})
  await sleep(180)
  assert('Story reduced motion keeps content visible and stops face animation',await page.eval("[...document.querySelectorAll('.story-reveal')].every(e=>getComputedStyle(e).opacity==='1'&&getComputedStyle(e).transform==='none')&&document.querySelector('.story-hero-face .hero-visual').dataset.motion==='paused'"))
  await page.call('Emulation.setEmulatedMedia',{features:[]})
  await page.call('Page.navigate',{url:'http://127.0.0.1:8000/'})
  for(let i=0;i<120;i++){if(await page.eval("document.querySelectorAll('[data-story-scene]').length===6 && document.querySelector('.story-connection')?.textContent.includes('Alphabet model ready')"))break;await sleep(100)}
  assert('Production bundle serves the story and same-origin model readiness',await page.eval("location.port==='8000' && document.querySelectorAll('[data-story-scene]').length===6 && document.querySelector('.story-connection')?.textContent.includes('Alphabet model ready') && window.__cameraCalls===0"))
  await screenshot('production-story.png')
  }
  await page.call('Page.navigate',{url:'http://127.0.0.1:5173/science'})
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('h1')"))break;await sleep(100)}
  assert('Existing science route remains available',await page.eval("!!document.querySelector('h1') && !document.body.innerText.includes('Unexpected Application Error') && window.__cameraCalls===0"))
  for (const width of [1366, 768, 390]) {
    await page.call('Emulation.setDeviceMetricsOverride',{width,height:1000,deviceScaleFactor:1,mobile:width===390})
    await page.eval('document.fonts.ready.then(()=>true)')
    const math = await page.eval(`({errors:document.querySelectorAll('[data-math-status="error"]').length,formulas:document.querySelectorAll('.math-formula').length,rendered:document.querySelectorAll('.math-formula .katex').length,mathml:document.querySelectorAll('math').length,overflow:document.documentElement.scrollWidth>innerWidth+1,font:document.fonts.check('16px KaTeX_Main'),bounds:[...document.querySelectorAll('.math-expression-display')].every(e=>{const r=e.getBoundingClientRect();return r.left>=-1&&r.right<=innerWidth+1})})`)
    assert('Science mathematics at '+width+' uses local math fonts, MathML, and contained equations',math.errors===0&&math.rendered>=math.formulas&&math.mathml>15&&!math.overflow&&math.font&&math.bounds,math)
    assert('Science header at '+width+' keeps visible navigation inside the viewport',await page.eval("[...document.querySelectorAll('.science-nav a')].filter(a=>a.getClientRects().length).every(a=>{const r=a.getBoundingClientRect();return r.left>=0&&r.right<=innerWidth+1})"))
    await page.eval("window.scrollTo({top:document.querySelector('#scene-math .sci-split').getBoundingClientRect().top+scrollY-100,behavior:'instant'})")
    await sleep(250)
    await screenshot('science-math-'+width+'.png')
    if(width===390){
      await page.eval("window.scrollTo({top:document.querySelector('#scene-math .sci-split > :nth-child(2)').getBoundingClientRect().top+scrollY-90,behavior:'instant'})")
      await screenshot('science-loss-mobile.png')
    }
    for(const tab of [0,1]){
      await page.eval(`document.querySelectorAll('#scene-architecture .sci-tab-btn')[${tab}].click()`)
      const stageCount=tab===0?7:4
      for(let stage=0;stage<stageCount;stage++){
        await page.eval(`document.querySelectorAll('#scene-architecture [style*="cursor: pointer"]')[${stage}].click()`)
        const state=await page.eval(`({errors:document.querySelectorAll('[data-math-status="error"]').length,tex:document.querySelector('#scene-architecture .math-expression-display')?.dataset.mathTex,overflow:document.documentElement.scrollWidth>innerWidth+1})`)
        assert('Architecture '+width+'/'+tab+'/'+stage+' typesets without errors or page overflow',state.errors===0&&!!state.tex&&!state.overflow,state)
        if(width===390&&tab===1&&stage===2){
          await page.eval("window.scrollTo({top:document.querySelector('#scene-architecture .math-box').getBoundingClientRect().top+scrollY-100,behavior:'instant'})")
          await screenshot('science-attention-mobile.png')
        }
      }
    }
  }
  await page.call('Emulation.setDeviceMetricsOverride',{width:1366,height:900,deviceScaleFactor:1,mobile:false})
  const interaction=await page.eval(`(async()=>{
    const value=()=>document.querySelector('[data-testid="cross-entropy-value"]').textContent;
    const initial=value();
    const slider=document.querySelector('[aria-label="Softmax temperature"]');
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(slider,'2');slider.dispatchEvent(new Event('input',{bubbles:true}));await new Promise(r=>setTimeout(r,60));const temperature=value();
    document.querySelector('#scene-math input[type="checkbox"]').click();await new Promise(r=>setTimeout(r,60));const unsmoothed=value();
    const target=[...document.querySelectorAll('#scene-math button')].find(b=>b.textContent.trim()==='E');target.click();await new Promise(r=>setTimeout(r,60));
    return {initial,temperature,unsmoothed,target:value()};
  })()`)
  assert('Temperature, label smoothing, and target controls still update mathematical results',new Set(Object.values(interaction)).size===4,interaction)
  await page.call('Page.navigate',{url:'http://127.0.0.1:8000/science#scene-math'})
  for(let i=0;i<100;i++){if(await page.eval("!!document.querySelector('#scene-math .katex')"))break;await sleep(100)}
  await page.eval('document.fonts.ready.then(()=>true)')
  assert('Production Science serves bundled math and accessible equations',await page.eval("document.querySelectorAll('#scene-math math').length>=8 && !document.querySelector('[data-math-status=error]') && document.fonts.check('16px KaTeX_Main')"))
  const exceptions = page.events.filter(event=>event.method==='Runtime.exceptionThrown').map(event=>event.params.exceptionDetails.exception?.description || event.params.exceptionDetails.text)
  assert('No uncaught browser exceptions',exceptions.length===0,exceptions)
  const failed = page.events.filter(event=>event.method==='Network.responseReceived' && event.params.response.status>=400).map(event=>({url:event.params.response.url,status:event.params.response.status}))
  const report={checks,exceptions,failedResponses:failed,screenshots:output,physicalCameraTested:false}
  await writeFile(path.join(output,'browser-report.json'),JSON.stringify(report,null,2))
  console.log(JSON.stringify({checksPassed:checks.filter(check=>check.passed).length,total:checks.length,exceptions,failedResponses:failed,report:path.join(output,'browser-report.json'),physicalCameraTested:false},null,2))
} catch(error) {
  if(page)try{await writeFile(path.join(output,'failure-state.txt'),await page.eval('document.body.innerText'));const {data}=await page.call('Page.captureScreenshot',{format:'png'});await writeFile(path.join(output,'failure.png'),Buffer.from(data,'base64'))}catch{}
  await writeFile(path.join(output,'browser-report.json'),JSON.stringify({checks,error:String(error)},null,2))
  console.error(error);process.exitCode=1
} finally {
  if(browser)try{await browser.call('Browser.close')}catch{}
  else chrome.kill()
}

