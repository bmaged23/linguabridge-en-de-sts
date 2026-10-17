class MicrophoneCapture extends AudioWorkletProcessor {
  constructor(){super();this.sum=0;this.count=0;this.phase=0;this.frame=new Float32Array(512);this.index=0;}
  process(inputs,outputs){
    const channel=inputs[0]?.[0] || new Float32Array(outputs[0]?.[0]?.length || 128);
    if(channel) for(const value of channel){
      this.sum+=value;this.count++;this.phase+=16000;
      if(this.phase>=sampleRate){
        this.phase-=sampleRate;this.frame[this.index++]=this.sum/this.count;this.sum=0;this.count=0;
        if(this.index===512){this.port.postMessage(this.frame.buffer,[this.frame.buffer]);this.frame=new Float32Array(512);this.index=0;}
      }
    }
    // Remain connected to the audio graph without monitoring microphone audio.
    for(const output of outputs) for(const channel of output) channel.fill(0);
    return true;
  }
}
registerProcessor('microphone-capture',MicrophoneCapture);
