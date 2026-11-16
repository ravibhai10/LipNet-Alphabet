// Software fixtures with genuine-video camera replay, not labelled speech data.
import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdtemp, readdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
const executable=process.env.LIPNET_CHROME_PATH??[process.env.PROGRAMFILES,process.env['PROGRAMFILES(X86)'],process.env.LOCALAPPDATA].filter(Boolean).map(p=>path.join(p,'Google/Chrome/Application/chrome.exe')).find(existsSync)
if(!executable)throw new Error('Set LIPNET_CHROME_PATH to the installed Chrome executable.')
const profile=await mkdtemp(path.resolve('.qa/word-collection-chrome-'))
const downloads=await mkdtemp(path.resolve('.qa/collection-fixtures-'))
const chrome=spawn(executable,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-timer-throttling','--disable-renderer-backgrounding','--remote-debugging-port=0','--user-data-dir='+profile,'--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+path.resolve('.qa/afternoon-paced-replay.y4m'),'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']})
class CDP {
  constructor(socket){this.socket=socket;this.next=0;this.pending=new Map();this.errors=[];socket.addEventListener('message',event=>{const d=JSON.parse(event.data);if(d.id){const p=this.pending.get(d.id);if(!p)return;clearTimeout(p.timer);this.pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result)}else if(d.method==='Runtime.exceptionThrown')this.errors.push(d.params)})}
  call(method,params={}){const id=++this.next;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('CDP timeout '+method)),30000);this.pending.set(id,{resolve,reject,timer});this.socket.send(JSON.stringify({id,method,params}))})}
  async eval(expression){const r=await this.call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value}
}
async function connect(url){const s=new WebSocket(url);await new Promise((resolve,reject)=>{s.addEventListener('open',resolve,{once:true});s.addEventListener('error',reject,{once:true})});return new CDP(s)}
const sleep=ms=>new Promise(r=>setTimeout(r,ms)),checks=[]
const check=(condition,name)=>{if(!condition)throw new Error(name);checks.push(name);console.log('PASS '+name)}
let browser,page
try {
  const address=await new Promise((resolve,reject)=>{let log='';const timer=setTimeout(()=>reject(new Error('Chrome start timeout')),30000);chrome.on('error',reject);chrome.stderr.on('data',b=>{log+=b;const m=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1])}})})
  browser=await connect(address)
  await browser.call('Browser.setDownloadBehavior',{behavior:'allow',downloadPath:downloads})
  const target=await(await fetch('http://'+new URL(address).host+'/json/new?about:blank',{method:'PUT'})).json();page=await connect(target.webSocketDebuggerUrl)
  await page.call('Page.enable');await page.call('Runtime.enable')
  await page.call('Page.addScriptToEvaluateOnNewDocument',{source:`window.__collectionQA={streams:[],posts:0};if(navigator.mediaDevices?.getUserMedia){const orig=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async(...args)=>{const s=await orig(...args);window.__collectionQA.streams.push(s);return s}}const f=window.fetch;window.fetch=(url,init)=>{if(init?.method==='POST')window.__collectionQA.posts++;const p=new URL(url,location.href).pathname;if(p==='/health'||p==='/api/health')return Promise.resolve(new Response(JSON.stringify({status:'ok',inference_ready:true,model_loaded:true})));if(p.includes('/api/word/')&&p.endsWith('/health'))return Promise.resolve(new Response(JSON.stringify({model_loaded:false,inference_ready:false,vocabulary:[]})));return f(url,init)}`})
  await page.call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false})
  const base=process.env.LIPNET_QA_BASE??'http://127.0.0.1:8001'
  await page.call('Page.navigate',{url:base+'/demo/collect'})
  const wait=async(expression,ms=20000)=>{const until=Date.now()+ms;while(Date.now()<until){if(await page.eval(expression))return true;await sleep(90)}return false}
  const click=label=>page.eval(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw new Error('Button unavailable: '+${JSON.stringify(label)});b.click()})()`)
  check(await wait('document.querySelector(".word-collection")!==null'),'collection route renders in existing workspace')
  check(await page.eval('document.querySelector("video").srcObject===null'),'collection camera starts off')
  check(await page.eval('[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==="Start camera").disabled'),'explicit recording consent required')
  await page.eval('document.querySelector(".wc-consent input").click()');await click('Start camera')
  check(await wait('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="Record Word"&&!b.disabled)',60000),'real face/ROI tracking enables collection')
  await click('Record Word')
  check(await wait('document.querySelector("progress").value>=9',7000),'capture accumulates fresh mouth frames')
  check(await page.eval('document.querySelector(".wc-fields select").disabled'),'partition/session settings locked during capture')
  await click('Finish Word')
  check(await wait('document.querySelectorAll(".wc-review img").length>0'),'review displays actual captured frames')
  check(await page.eval('[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==="Save labelled clip").disabled'),'spoken-word confirmation required before save')
  await page.eval('document.querySelector(".wc-prompt input[type=checkbox]").click()')
  await page.eval('(()=>{const b=[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==="Save labelled clip");b.click();b.click()})()')
  check(await wait('document.querySelectorAll(".wc-saved li").length===1'),'duplicate Save creates one durable recording')
  check(await page.eval('document.body.innerText.includes("Saved choose locally")'),'successful local save is explicit')
  const desktop=await page.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await writeFile('.qa/word-collection-desktop.png',Buffer.from(desktop.data,'base64'))
  await click('Stop camera')
  check(await page.eval('window.__collectionQA.streams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))'),'Stop releases collection camera tracks')
  await page.call('Page.reload')
  check(await wait('document.querySelectorAll(".wc-saved li").length===1'),'IndexedDB recording survives reload')
  check(await page.eval('document.querySelector(".wc-counts button").innerText.includes("1 / 15")'),'progress derives from saved recording count')
  await click('Export training bundle')
  let file
  for(let i=0;i<100;i++){file=(await readdir(downloads)).find(n=>n.endsWith('.json'));if(file)break;await sleep(100)}
  check(!!file,'download contains an actual recording bundle')
  const bundle=JSON.parse(await readFile(path.join(downloads,file),'utf8'))
  check(bundle.format==='lipnet-word-collection-v1'&&bundle.clips.length===1&&bundle.clips[0].frames.length>=8,'bundle records expected schema and captured data')
  check(bundle.clips[0].label==='choose'&&bundle.clips[0].label_confirmed&&bundle.clips[0].landmarker_sha256.length===64,'bundle preserves user label and model fingerprint')
  bundle.qa_fixture=true;bundle.fixture_note='Chrome software test with AFTERNOON replay labelled choose solely to test collection; never training data.'
  await writeFile(path.join(downloads,file),JSON.stringify(bundle))
  await page.eval('(()=>{const s=document.querySelector(".wc-fields select");Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,"value").set.call(s,"valid");s.dispatchEvent(new Event("change",{bubbles:true}))})()')
  check(await wait('document.querySelector(".wc-fields input:last-of-type")!==null && document.querySelector(".wc-fields select").value==="valid"'),'validation partition selectable with camera stopped')
  check(await page.eval('document.querySelectorAll(".wc-fields input")[1].value.startsWith("valid-")'),'changing partition starts a separate session ID')
  check(await page.eval('document.querySelector(".wc-counts button").innerText.includes("0 / 5")'),'validation progress excludes training clips')
  check(await page.eval('window.__collectionQA.posts===0'),'collection sends no predictions or recording uploads')
  await page.call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true})
  check(await page.eval('document.documentElement.scrollWidth<=innerWidth+1'),'mobile collection has no horizontal overflow')
  const {data}=await page.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await writeFile('.qa/word-collection-mobile.png',Buffer.from(data,'base64'))
  await page.eval('document.querySelector(".wc-consent input").click()');await click('Start camera')
  check(await wait('[...document.querySelectorAll("button")].some(b=>b.textContent.trim()==="Record Word"&&!b.disabled)',60000),'collection camera can restart after export')
  await click('Record Word')
  await page.eval(`document.querySelector('a[href="/demo/live"]').click()`)
  check(await wait('document.querySelector(".ws-live")!==null'),'back link returns to existing live recognition')
  check(await page.eval('window.__collectionQA.streams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))'),'navigation during recording releases every camera track')
  check(page.errors.length===0,'no uncaught collection browser errors')
  await writeFile('.qa/word-collection-browser.json',JSON.stringify({checks_passed:checks.length,checks,fixture_only:true,bundle:path.relative(process.cwd(),path.join(downloads,file)),uncaught_errors:page.errors},null,2))
  console.log(checks.length+' collection browser checks passed')
}catch(error){await writeFile('.qa/word-collection-browser-failure.json',JSON.stringify({error:String(error),checks,exceptions:page?.errors,text:page?await page.eval('document.body.innerText').catch(()=>null):null},null,2));throw error}
finally{if(page)page.socket.close();if(browser){try{await browser.call('Browser.close')}catch{}browser.socket.close()}if(chrome.exitCode===null)chrome.kill()}
