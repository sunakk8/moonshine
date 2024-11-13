import Moonshine from "./moonshine.js"

function setTranscription(text) {
    document.getElementById("transcription").innerHTML = text
}

async function startRecording() {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    const audioChunks = [];
  
    mediaRecorder.ondataavailable = event => {
      audioChunks.push(event.data);
    };
  
    mediaRecorder.start();
    console.log("Recording started");
  
    return new Promise(resolve => {
      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        console.log("Recording stopped");
        resolve(audioBlob);
      };
  
      setTimeout(() => mediaRecorder.stop(), 10000);
    });
  }

window.onload = (event) => {
    var moonshine = new Moonshine("tiny")
    moonshine.loadModel().then(() => {
        setTranscription("")
    });

    upload.onchange = function(e) {
        var sound = document.getElementById("sound")
        sound.src = URL.createObjectURL(this.files[0])
    }

    transcribe.onclick = async function(e) {
        var sound = document.getElementById("sound")
        if (sound.src) {
            const audioCTX = new AudioContext({
                sampleRate: 16000,
            });
            let file = await fetch(sound.src).then(r => r.blob())
            let data = await file.arrayBuffer()
            let decoded = await audioCTX.decodeAudioData(data);
            let floatArray = new Float32Array(decoded.length)
            decoded.copyFromChannel(floatArray, 0)
            moonshine.generate(floatArray).then((r) => {
                document.getElementById("transcription").innerText = r
            })
        }
    }

    startRecord.onclick = function(e) {
        document.getElementById("startRecord").style = "display: none;"
        document.getElementById("stopRecord").style = "display: block;"

        // fired when recording hits the time limit
        startRecording().then(audioBlob => {
            document.getElementById("startRecord").style = "display: block;"
            document.getElementById("stopRecord").style = "display: none;"
            console.log("Recorded Blob:", audioBlob);
            var sound = document.getElementById("sound")
            sound.src = URL.createObjectURL(audioBlob)
        });
    }
    stopRecord.onclick = function(e) {
        document.getElementById("startRecord").style = "display: block;"
        document.getElementById("stopRecord").style = "display: none;"
    }
};

