import test from 'node:test'
import assert from 'node:assert/strict'
import { InferenceSocket, inferenceSocketUrl } from '../src/lib/inferenceSocket.ts'

test('socket URLs preserve ws schemes and relative configured endpoints', () => {
  assert.equal(inferenceSocketUrl('http://127.0.0.1:8000','http://localhost:5173/'),'ws://127.0.0.1:8000/ws/inference')
  assert.equal(inferenceSocketUrl('', 'https://example.org/', '/ws'),'wss://example.org/ws/inference')
  assert.equal(inferenceSocketUrl('', 'https://example.org/', 'wss://api.example.org/ws/inference'),'wss://api.example.org/ws/inference')
})
test('duplicate connect and intentional disconnect never create a reconnect', t => {
  t.mock.timers.enable({apis:['setTimeout','setInterval']})
  const sockets=[], states=[]
  const client=new InferenceSocket('ws://example.org',state=>states.push(state),()=>{},()=> {
    const socket={readyState:0, bufferedAmount:0, close(){this.readyState=3;this.onclose?.()},send(){}}
    sockets.push(socket); return socket
  })
  client.connect(); client.connect()
  assert.equal(sockets.length,1)
  const staleOpen=sockets[0].onopen
  client.disconnect(); staleOpen()
  t.mock.timers.tick(90000)
  assert.equal(sockets.length,1)
  assert.equal(states.at(-1),'idle')
})
test('retry attempts and outgoing buffering are bounded', t => {
  t.mock.timers.enable({apis:['setTimeout','setInterval']})
  const sockets=[]
  const client=new InferenceSocket('ws://example.org',()=>{},()=>{},()=> {
    const socket={readyState:0,bufferedAmount:0,close(){},send(){}}
    sockets.push(socket); return socket
  })
  client.connect()
  for(let i=0;i<10;i++){sockets.at(-1).onclose?.();t.mock.timers.tick(20000)}
  assert.equal(sockets.length,6)
  assert.equal(client.sendFrame('fixture'),false)
  client.disconnect();client.connect()
  const socket=sockets.at(-1);socket.readyState=1;socket.onopen()
  socket.bufferedAmount=129000
  assert.equal(client.sendFrame('fixture'),false)
  socket.bufferedAmount=0
  assert.equal(client.sendFrame('fixture'),true)
  client.disconnect()
})
