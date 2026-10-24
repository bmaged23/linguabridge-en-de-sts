"""Bursty capture must retain samples without hiding sustained congestion."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_microphone_bursts_drain_in_order_and_congestion_is_bounded():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for browser transport checks')
    script = Path(__file__).resolve().parents[1] / 'web/microphone-transport.js'
    code = r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
let timer=null,overflows=0;
const scope={setTimeout:fn=>{timer=fn;return 1;},clearTimeout:()=>{timer=null;}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),scope);
const sent=[],socket={readyState:1,bufferedAmount:0,send(frame){sent.push(frame.id);this.bufferedAmount+=2048;}};
const options={maxBufferedBytes:65536,maxPendingFrames:64,drainIntervalMs:8,onOverflow:()=>overflows++};
const transport=new scope.BridgeMicrophoneTransport(socket,options);
for(let i=0;i<80;i++)transport.send({id:i});
assert.equal(overflows,0);assert.equal(sent.length,32);
while(timer){socket.bufferedAmount=0;const next=timer;timer=null;next();}
assert.deepEqual(sent,Array.from({length:80},(_,i)=>i));
assert.equal(transport.pending.length,0);
socket.bufferedAmount=65536;
for(let i=0;i<65;i++)transport.send({id:80+i});
assert.equal(overflows,1);assert.equal(transport.pending.length,0);assert.equal(timer,null);
transport.send({id:999});assert.equal(sent.length,80);
const next=new scope.BridgeMicrophoneTransport(socket,options);
next.send({id:1});assert(timer);next.close();assert.equal(timer,null);
assert.equal(next.pending.length,0);
'''
    subprocess.run([node, '-e', code, str(script)], check=True, capture_output=True, text=True)
