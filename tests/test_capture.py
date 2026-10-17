"""Microphone framing must continue through an empty/silent input quantum."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_capture_emits_clocked_silence_and_resamples_voice():
    node=shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the AudioWorklet regression check')
    script=Path(__file__).resolve().parents[1]/'web/capture.js'
    code=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const sent=[];let Processor;
class Base {constructor(){this.port={postMessage:buffer=>sent.push(new Float32Array(buffer).slice())};}}
const sandbox={AudioWorkletProcessor:Base,sampleRate:48000,Float32Array,
 registerProcessor:(name,cls)=>{Processor=cls;}};
vm.runInNewContext(fs.readFileSync(process.argv[1],'utf8'),sandbox);
const worker=new Processor();
for(let i=0;i<12;i++)worker.process([[]],[[new Float32Array(128)]]);
assert.equal(sent.length,1);assert.equal(sent[0].length,512);
assert(sent[0].every(value=>value===0));
for(let i=0;i<12;i++)worker.process([[new Float32Array(128).fill(.2)]],[[new Float32Array(128)]]);
assert.equal(sent.length,2);assert(sent[1].every(value=>Math.abs(value-.2)<1e-6));
'''
    subprocess.run([node,'-e',code,str(script)],check=True,capture_output=True,text=True)


def test_echo_guard_protects_playback_and_tail_without_blocking_idle_speech():
    node=shutil.which('node')
    if not node:pytest.skip('Node is needed for browser audio policy checks')
    script=Path(__file__).resolve().parents[1]/'web/playback-policy.js'
    code=r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={};vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
const gate=context.BridgePlaybackPolicy.suppressMicrophone;
const base={mode:'conversation',translationGuard:true,echoGuard:true,allowInterruptions:false,playing:false,queued:0,resumeAt:0,now:100};
assert.equal(gate(base),false);
assert.equal(gate({...base,playing:true}),true);
assert.equal(gate({...base,queued:1}),true);
assert.equal(gate({...base,resumeAt:200}),true);
assert.equal(gate({...base,resumeAt:100}),false);
assert.equal(gate({...base,playing:true,allowInterruptions:true}),false);
assert.equal(gate({...base,playing:true,echoGuard:false}),false);
assert.equal(gate({...base,mode:'translation',playing:true,allowInterruptions:true}),true);
assert.equal(gate({...base,mode:'translation',translationGuard:false,playing:true,allowInterruptions:true}),false);
'''
    subprocess.run([node,'-e',code,str(script)],check=True,capture_output=True,text=True)
