import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import requests
import subprocess

from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition

app = Flask(__name__)

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",
    run_opts={"device": "cpu"}
)


@app.route("/verify", methods=["POST"])
def verify():
    data = request.get_json()
    audio_url = data["audio_url"]

    # Descargar audio
    response = requests.get(audio_url)
    with open("temp.oga", "wb") as f:
        f.write(response.content)

    # Convertir a WAV 16kHz mono
    subprocess.run([
        "ffmpeg", "-y",
        "-i", "temp.oga",
        "-ac", "1",
        "-ar", "16000",
        "temp.wav"
    ], check=True)

    # Verificar voz
    score, prediction = verification.verify_files(
        "authorized.wav",
        "temp.wav"
    )

    if bool(prediction):
        return jsonify({"resultado": "valido"})
    else:
        return jsonify({"resultado": "invalido"})


app.run(host="0.0.0.0", port=5000)