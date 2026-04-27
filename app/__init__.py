from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
import os
import requests
import subprocess


verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",
    run_opts={"device": "cpu"}
)


def create_app():
    app = Flask(__name__)

    @app.route("/verify", methods=["POST"])
    def verify():
        data = request.json
        audio_url = data.get("audio_url")

        response = requests.get(audio_url)
        with open("temp.oga", "wb") as f:
            f.write(response.content)

        subprocess.run([
            "ffmpeg", "-y",
            "-i", "temp.oga",
            "-ac", "1",
            "-ar", "16000",
            "temp.wav"
        ], check=True)

        score, prediction = verification.verify_files(
            "authorized.wav",
            "temp.wav"
        )

        if bool(prediction):
            return jsonify({"resultado": "valido"})
        else:
            return jsonify({"resultado": "invalido"})

    return app