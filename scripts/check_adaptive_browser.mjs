// Software UI tests: explicit API fixtures. These are not live model or accuracy evidence.
import {spawn} from 'node:child_process'
import {existsSync} from 'node:fs'
import {mkdtemp,writeFile} from 'node:fs/promises'
import path from 'node:path'
const base=process.env.LIPNET_QA_BASE??'http://127.0.0.1:8018'
const output=path.resolve('.qa/adaptive_20261116')
const chromePath=[process.env.PROGRAMFILES,process.env['PROGRAMFILES(X86)'],process.env.LOCALAPPDATA].filter(Boolean).map(p=>path.join(p,'Google/Chrome/Application/chrome.exe')).find(existsSync)
if(!chromePath)throw new Error('Chrome unavailable')
const profile=await mkdtemp(path.join(output,'browser-'))
const chrome=spawn(chromePath,['--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--disable-background-timer-throttling','--disable-renderer-backgrounding','--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+path.resolve('.qa/afternoon-paced-replay.y4m'),'--remote-debugging-port=0','--user-data-dir='+profile,'about:blank'],{windowsHide:true,stdio:['ignore','ignore','pipe']})
class CDP{
  constructor(ws){this.ws=ws;this.next=0;this.pending=new Map();this.errors=[];ws.onmessage=e=>{const d=JSON.parse(e.data);if(d.id){const p=this.pending.get(d.id);if(p){clearTimeout(p.timer);this.pending.delete(d.id);d.error?p.reject(new Error(JSON.stringify(d.error))):p.resolve(d.result)}}else if(d.method==='Runtime.exceptionThrown')this.errors.push(d.params)}}
  call(method,params={}){const id=++this.next;return new Promise((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('CDP timeout '+method)),30000);this.pending.set(id,{resolve,reject,timer});this.ws.send(JSON.stringify({id,method,params}))})}
  async eval(expression){const r=await this.call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true,userGesture:true});if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));return r.result.value}
}
async function connect(url){const ws=new WebSocket(url);await new Promise((resolve,reject)=>{ws.onopen=resolve;ws.onerror=reject});return new CDP(ws)}
const checks=[];let browser,page
const check=(value,name)=>{if(!value)throw new Error(name);checks.push(name);console.log('PASS '+name)}
const fixture=`
window.__adaptiveQA={feedback:0,withdraw:0,review:0};
const realFetch=window.fetch.bind(window);const q=window.__adaptiveQA;
const record={id:'ui-fixture',task:'isolated_word',speaker:'qa-speaker',session:'qa-session',status:'PENDING_REVIEW',label:null,verdict:null,created:'2026-11-16T10:00:00Z',frame_count:10,captured_fps:10,duration_ms:900,warnings:[],media_available:true,capture_kind:'webcam_replay',prediction:{prediction:'hello',candidate:'hello',confidence:.96,model_id:'EXPLICIT_UI_FIXTURE',model_version:'fixture-version',latency_ms:123,status:'PREDICTED',top_k:[{label:'hello',score:.96},{label:'stop',score:.03}]}};
window.fetch=async(url,init={})=>{
const p=new URL(url,location.href).pathname;const json=(v,status=200)=>new Response(JSON.stringify(v),{status,headers:{'Content-Type':'application/json'}});
if(p==='/health'||p==='/api/health')return json({status:'ok',inference_ready:false});
if(!p.startsWith('/api/adaptive'))return realFetch(url,init);
if(!init.headers?.Authorization)return json({detail:'Private key required'},401);
if(p.endsWith('/config'))return json({tasks:{isolated_word:['hello','stop'],alphabet_recognition:['A','B']},correction_labels:{isolated_word:['hello','stop','help'],alphabet_recognition:['A','B']},role:'reviewer',recording_consent:'Explicit fixture recording and training consent'});
if(p.endsWith('/stats'))return json({total:1,confirmed:record.verdict==='correct'?1:0,corrected:record.verdict==='incorrect'?1:0,unique_speakers:1,statuses:{[record.status]:1},classes:{},growth:{'2026-11-16':1},training_eligible:record.status==='APPROVED'?1:0,versions:[]});
if(p.endsWith('/records')||p.endsWith('/hard-examples'))return json({records:[record]});
if(p.endsWith('/models'))return json({models:[{id:'EXPLICIT_UI_FIXTURE',task:'isolated_word',version:'fixture-version',training_dataset_version:'fixture-version',deployment_status:'preserved_research_baseline',validation_report:null,test_report:null,cpu_latency_ms:null}],champions:{isolated_word:{model:'EXPLICIT_UI_FIXTURE',previous:null}}});
if(p.endsWith('/captures')){q.captureBody=JSON.parse(init.body);Object.assign(record,{id:q.captureBody.id,status:'PENDING_REVIEW',verdict:null,label:null,media_available:true});return json(record)}
if(p.includes('/frames/'))return realFetch('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9Zl1sAAAAASUVORK5CYII=');
if(p.endsWith('/feedback')){q.feedback++;Object.assign(record,JSON.parse(init.body));return json(record)}
if(p.endsWith('/withdraw')){q.withdraw++;record.status='WITHDRAWN';record.media_available=false;return json({status:'WITHDRAWN'})}
if(p.endsWith('/review')){q.review++;record.status=JSON.parse(init.body).decision;return json(record)}
return json(record);
};`
try{
 const address=await new Promise((resolve,reject)=>{let log='';const timeout=setTimeout(()=>reject(new Error('Chrome startup timeout')),30000);chrome.on('error',reject);chrome.stderr.on('data',b=>{log+=b;const match=log.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(match){clearTimeout(timeout);resolve(match[1])}})})
 browser=await connect(address);const target=await(await fetch('http://'+new URL(address).host+'/json/new?about:blank',{method:'PUT'})).json();page=await connect(target.webSocketDebuggerUrl)
 await page.call('Page.enable');await page.call('Runtime.enable');await page.call('Page.addScriptToEvaluateOnNewDocument',{source:fixture})
 const wait=async(expression)=>{for(let i=0;i<180;i++){if(await page.eval(expression))return true;await new Promise(r=>setTimeout(r,60))}return false}
 const click=async text=>page.eval(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(text)});if(!b||b.disabled)throw new Error('Unavailable button '+${JSON.stringify(text)});b.click()})()`)
 const input=async(selector,value)=>page.eval(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('input',{bubbles:true}))})()`)
 await page.call('Emulation.setDeviceMetricsOverride',{width:1440,height:1100,deviceScaleFactor:1,mobile:false})
 await page.call('Page.navigate',{url:base+'/demo/feedback'})
 check(await wait(`document.body.innerText.includes('Adaptive speech lab')`),'adaptive route renders')
 check(await page.eval(`!document.body.innerText.includes('Was this prediction correct?')`),'unconnected workspace does not invent a prediction')
 await input('.af-access input','q'.repeat(32));await click('Connect')
 check(await wait(`document.body.innerText.includes('reviewer access')`),'access key connects role-scoped workspace')
 await click('Dataset Feedback Studio');await click('Review source')
 check(await wait(`document.body.innerText.includes('Was this prediction correct?')`),'saved source opens feedback panel')
 check(await page.eval(`document.body.innerText.includes('EXPLICIT_UI_FIXTURE')&&document.body.innerText.includes('123 ms')`),'feedback panel shows identity and latency')
 check(await page.eval(`!!document.querySelector('input[type=range]')&&!!document.querySelector('img.af-source')`),'source preview and frame timeline present')
 await click('No, incorrect');check(await wait(`document.body.innerText.includes('Actual spoken label')`),'incorrect action requests explicit correction')
 check(await page.eval(`!![...document.querySelectorAll('button')].find(b=>b.textContent==='Save correction'&&b.disabled)`),'missing correction blocks submission')
 await page.eval(`(()=>{const e=[...document.querySelectorAll('select')].find(e=>[...e.options].some(o=>o.value==='UNKNOWN'));e.value='stop';e.dispatchEvent(new Event('change',{bubbles:true}))})()`)
 await click('Save correction');check(await wait(`document.body.innerText.includes('Human feedback saved')`),'corrected feedback visibly saved')
 check(await page.eval(`window.__adaptiveQA.feedback===1&&!![...document.querySelectorAll('button')].find(b=>b.textContent==='Save correction'&&b.disabled)`),'duplicate feedback click blocked')
 await page.eval(`(()=>{const e=[...document.querySelectorAll('input')].find(e=>e.parentElement.textContent.includes('Review reason'));Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,'Inspected software fixture');e.dispatchEvent(new Event('input',{bubbles:true}))})()`)
 await click('APPROVED');check(await wait(`document.body.innerText.includes('Dataset status: APPROVED')`),'review decision updates dataset status')
 await click('Undo / withdraw recording');check(await wait(`document.body.innerText.includes('Recording withdrawn')`),'withdrawal acknowledged')
 check(await page.eval(`!document.querySelector('img.af-source')`),'withdrawal clears private preview')
 await click('Retry / next utterance');await click('Start camera')
 check(await wait(`document.body.innerText.includes('One face; mouth landmarks detected.')`),'real MediaPipe tracks a genuine prerecorded camera source')
 await input('input[placeholder="Stable pseudonymous ID"]','qa-replay-only')
 await page.eval(`document.querySelector('.af-check input').click()`)
 await click('Start utterance')
 check(await wait(`document.querySelector('.af-page progress')?.value>=10`),'bounded capture collects chronological source frames')
 await click('Finish utterance')
 check(await wait(`!!document.querySelector('video.af-source[src^="data:video/"]')`),'complete MediaRecorder clip available for native replay')
 await click('Predict reviewed recording')
 check(await wait(`!!window.__adaptiveQA.captureBody`),'source recording submits with feedback prediction request')
 check(await page.eval(`window.__adaptiveQA.captureBody.frames.length>=8&&window.__adaptiveQA.captureBody.frames.every(f=>f.startsWith('data:image/png;base64,'))&&window.__adaptiveQA.captureBody.original_video.startsWith('data:video/webm;base64,')`),'lossless source PNGs and original bounded video travel together')
 check(await page.eval(`window.__adaptiveQA.captureBody.timestamps_ms.every((t,i,a)=>i===0||t>a[i-1])`),'captured timestamps are strictly chronological')
 await click('Models & training');check(await wait(`document.body.innerText.includes('Evidence & current champions')`),'model comparison view renders')
 check(await page.eval(`document.body.innerText.includes('not measured')&&!document.body.innerText.includes('95.00%')`),'missing evaluation stays unmeasured')
 await click('Examiner walkthrough');check(await wait(`document.body.innerText.includes('External examiner walkthrough')`),'examiner workflow renders')
 await click('Next step');check(await page.eval(`document.querySelectorAll('[aria-current=step]').length===1`),'walkthrough tracks actual selected step')
 await page.call('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:true})
 for(const tab of ['Prediction & feedback','Dataset Feedback Studio','Models & training','Examiner walkthrough']){await click(tab);check(await page.eval('document.documentElement.scrollWidth<=innerWidth+1'),'no mobile overflow '+tab)}
 const {data}=await page.call('Page.captureScreenshot',{format:'png',captureBeyondViewport:true});await writeFile(path.join(output,'adaptive-mobile.png'),Buffer.from(data,'base64'))
 await click('Lock workspace');check(await wait(`!document.body.innerText.includes('reviewer access')`),'locking clears access and dataset views')
 await input('.af-access input','z'.repeat(32));await click('Connect')
 check(await wait(`document.body.innerText.includes('reviewer access')`),'a new access session can connect after locking')
 check(await page.eval(`!document.querySelector('.af-check input').checked&&document.querySelector('input[placeholder="Stable pseudonymous ID"]').value===''`),'locking clears recording consent and prior speaker identity')
 check(page.errors.length===0,'no uncaught browser exceptions')
 await writeFile(path.join(output,'browser_checks.json'),JSON.stringify({scope:'UI integration with explicit API and preview fixtures, plus real MediaPipe/MediaRecorder on a genuine prerecorded AFTERNOON camera replay. Not recognition or physical webcam evidence.',checks_passed:checks.length,checks,uncaught_errors:page.errors},null,2))
 console.log(checks.length+' checks passed')
}catch(error){await writeFile(path.join(output,'browser_failure.json'),JSON.stringify({error:String(error),checks,body:page?await page.eval('document.body.innerText').catch(()=>null):null,exceptions:page?.errors},null,2));throw error}
finally{if(page)page.ws.close();if(browser){try{await browser.call('Browser.close')}catch{}browser.ws.close()}if(chrome.exitCode===null)chrome.kill()}
