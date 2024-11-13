import argparse
import os
import sys
import time
import numpy as np
from queue import Queue
from silero_vad import VADIterator, load_silero_vad
from sounddevice import InputStream
from tokenizers import Tokenizer
from flask import Flask, render_template, Response
import threading
import subprocess

# Local import of Moonshine ONNX model.
MOONSHINE_DEMO_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(MOONSHINE_DEMO_DIR, ".."))

from onnx_model import MoonshineOnnxModel

SAMPLING_RATE = 16000
CHUNK_SIZE = 512  # Silero VAD requirement with sampling rate 16000.
LOOKBACK_CHUNKS = 5
MAX_LINE_LENGTH = 80
MAX_SPEECH_SECS = 15
MIN_REFRESH_SECS = 0.2

# Flask app
app = Flask(__name__)
transcription_queue = Queue()

class Transcriber(object):
    def __init__(self, model_name, rate=16000):
        if rate != 16000:
            raise ValueError("Moonshine supports sampling rate 16000 Hz.")
        self.model = MoonshineOnnxModel(model_name=model_name)
        self.rate = rate
        tokenizer_path = os.path.join(
            MOONSHINE_DEMO_DIR, "..", "assets", "tokenizer.json"
        )
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        self.inference_secs = 0
        self.number_inferences = 0
        self.speech_secs = 0
        self.__call__(np.zeros(int(rate), dtype=np.float32))  # Warmup.

    def __call__(self, speech):
        """Returns string containing Moonshine transcription of speech."""
        self.number_inferences += 1
        self.speech_secs += len(speech) / self.rate
        start_time = time.time()

        tokens = self.model.generate(speech[np.newaxis, :].astype(np.float32))
        text = self.tokenizer.decode_batch(tokens)[0]

        self.inference_secs += time.time() - start_time
        return text


def create_input_callback(q):
    """Callback method for sounddevice InputStream."""

    def input_callback(data, frames, time, status):
        if status:
            print(status)
        q.put((data.copy().flatten(), status))

    return input_callback


def end_recording(speech, do_print=True):
    """Transcribes, prints and caches the caption then clears speech buffer."""
    text = transcribe(speech)
    if do_print:
        print_captions(text)
    transcription_queue.put(text)  # Add the new caption to the queue immediately.
    caption_cache.append(text)
    speech *= 0.0


def print_captions(text):
    """Prints right justified on same line, prepending cached captions."""
    if len(text) < MAX_LINE_LENGTH:
        for caption in caption_cache[::-1]:
            text = caption + " " + text
            if len(text) > MAX_LINE_LENGTH:
                break
    if len(text) > MAX_LINE_LENGTH:
        text = text[-MAX_LINE_LENGTH:]
    else:
        text = " " * (MAX_LINE_LENGTH - len(text)) + text
    print("\r" + (" " * MAX_LINE_LENGTH) + "\r" + text, end="", flush=True)


def soft_reset(vad_iterator):
    """Soft resets Silero VADIterator without affecting VAD model state."""
    vad_iterator.triggered = False
    vad_iterator.temp_end = 0
    vad_iterator.current_sample = 0


@app.route('/')
def index():
    return render_template('index.html')


def generate():
    while True:
        if not transcription_queue.empty():
            transcription = transcription_queue.get()
            yield f"data: {transcription}\n\n"


@app.route('/stream')
def stream():
    return Response(generate(), mimetype='text/event-stream')


def run_flask():
    app.run(debug=True, use_reloader=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="live_captions",
        description="Live captioning demo of Moonshine models",
    )
    parser.add_argument(
        "--model_name",
        help="Model to run the demo with",
        default="moonshine/base",
        choices=["moonshine/base", "moonshine/tiny"],
    )
    args = parser.parse_args()
    model_name = args.model_name
    print(f"Loading Moonshine model '{model_name}' (using ONNX runtime) ...")
    transcribe = Transcriber(model_name=model_name, rate=SAMPLING_RATE)

    vad_model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(
        model=vad_model,
        sampling_rate=SAMPLING_RATE,
        threshold=0.5,
        min_silence_duration_ms=300,
    )

    q = Queue()
    stream = InputStream(
        samplerate=SAMPLING_RATE,
        channels=1,
        blocksize=CHUNK_SIZE,
        dtype=np.float32,
        callback=create_input_callback(q),
    )
    stream.start()

    caption_cache = []
    lookback_size = LOOKBACK_CHUNKS * CHUNK_SIZE
    speech = np.empty(0, dtype=np.float32)

    recording = False

    print("Press Ctrl+C to quit live captions.\n")

    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    with stream:
        print_captions("Ready...")
        try:
            while True:
                chunk, status = q.get()
                if status:
                    print(status)

                speech = np.concatenate((speech, chunk))
                if not recording:
                    speech = speech[-lookback_size:]

                speech_dict = vad_iterator(chunk)
                if speech_dict:
                    if "start" in speech_dict and not recording:
                        recording = True
                        start_time = time.time()

                    if "end" in speech_dict and recording:
                        recording = False
                        end_recording(speech)

                elif recording:
                    if (len(speech) / SAMPLING_RATE) > MAX_SPEECH_SECS:
                        recording = False
                        end_recording(speech)
                        soft_reset(vad_iterator)

                    if (time.time() - start_time) > MIN_REFRESH_SECS:
                        print_captions(transcribe(speech))
                        start_time = time.time()

        except KeyboardInterrupt:
            stream.close()

            if recording:
                while not q.empty():
                    chunk, _ = q.get()
                    speech = np.concatenate((speech, chunk))
                end_recording(speech, do_print=False)

            print(f"""

             model_name :  {model_name}
       MIN_REFRESH_SECS :  {MIN_REFRESH_SECS}s

      number inferences :  {transcribe.number_inferences}
    mean inference time :  {(transcribe.inference_secs / transcribe.number_inferences):.2f}s
  model realtime factor :  {(transcribe.speech_secs / transcribe.inference_secs):0.2f}x
""")
            if caption_cache:
                print(f"Cached captions.\n{' '.join(caption_cache)}")
