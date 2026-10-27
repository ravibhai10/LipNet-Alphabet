import test from 'node:test'
import assert from 'node:assert/strict'
import { VisualActivityDetector, PredictionStabilizer } from '../src/live-demo/stabilization.ts'
import { mouthAlignment } from '../src/live-demo/mouthUtils.ts'

const prediction = (a=.8, b=.2) => ({ letter: a >= b ? 'A' : 'B', confidence: Math.max(a,b),
  top3: [], probabilities: [{letter:'A',prob:a},{letter:'B',prob:b}] })

test('a static open or closed mouth does not trigger speaking', () => {
  for (const aperture of [0, .4]) {
    const vad = new VisualActivityDetector()
    for (let i=0;i<60;i++) assert.equal(vad.update(aperture,i*67).speaking,false)
  }
})
test('mouth movement triggers activity and silence and timing gaps clear it', () => {
  const vad = new VisualActivityDetector()
  let result
  for (let i=0;i<12;i++) result=vad.update(i%2 ? .2 : .05,i*67)
  assert.equal(result.speaking,true)
  for (let i=12;i<40;i++) result=vad.update(.1,i*67)
  assert.equal(result.speaking,false)
  assert.equal(vad.update(.2,10000).speaking,false)
  vad.reset()
  assert.equal(vad.update(.3,1).speaking,false)
})
test('EMA uses the full real vector and requires two consistent windows', () => {
  const smoother=new PredictionStabilizer(.6,2)
  assert.equal(smoother.update(prediction()).prediction,null)
  const result=smoother.update(prediction(.6,.4))
  assert.equal(result.confirmed,true)
  assert.ok(Math.abs(result.prediction.confidence-.68)<1e-12)
  assert.equal(result.prediction.stabilization.method,'ema')
  assert.ok(Math.abs(result.prediction.probabilities.reduce((s,p)=>s+p.prob,0)-1)<1e-12)
})
test('changed top prediction needs a new pair of windows', () => {
  const smoother=new PredictionStabilizer(1,2)
  smoother.update(prediction())
  assert.equal(smoother.update(prediction(.1,.9)).prediction,null)
  assert.equal(smoother.update(prediction(.1,.9)).prediction.letter,'B')
  smoother.reset()
  assert.equal(smoother.update(prediction(.1,.9)).prediction,null)
})
test('older APIs use consensus without fabricating missing probabilities', () => {
  const smoother=new PredictionStabilizer()
  const raw={letter:'A',confidence:.7,top3:[{letter:'A',prob:.7}]}
  assert.equal(smoother.update(raw).prediction,null)
  const result=smoother.update(raw).prediction
  assert.equal(result.confidence,.7)
  assert.equal(result.probabilities,undefined)
  assert.equal(result.stabilization.method,'consensus')
})
test('vocabulary changes cannot reuse prior confirmations', () => {
  const smoother=new PredictionStabilizer()
  smoother.update(prediction())
  const changed=prediction(); changed.probabilities[1].letter='C'
  assert.equal(smoother.update(changed).prediction,null)
  assert.equal(smoother.update(changed).confirmed,true)
})

test('alignment rejects invalid and off-image mouth coordinates', () => {
  const points=Array.from({length:468},()=>({x:.5,y:.5}))
  points[61]={x:.4,y:.6};points[291]={x:.6,y:.6};points[13]={x:.5,y:.58};points[14]={x:.5,y:.62}
  points[33]={x:.3,y:.3};points[263]={x:.7,y:.3}
  assert.ok(mouthAlignment(points,640,480))
  points[13].x=NaN
  assert.equal(mouthAlignment(points,640,480),null)
  points[13]={x:.1,y:.05};points[14]={x:.1,y:.06}
  assert.equal(mouthAlignment(points,640,480),null)
})
