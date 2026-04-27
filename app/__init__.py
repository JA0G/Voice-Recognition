import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
import requests
import subprocess

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
AUTHORIZED_FOLDER = os.path.join(BASE_DIR, "authorized_users")
THRESHOLD = 0.75

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",
    run_opts={"device": "cpu"}
)

def find_authorized_user(temp_audio_path):
    if not os.path.exists(AUTHORIZED_FOLDER):
        return None

    for filename in os.listdir(AUTHORIZED_FOLDER):
        if filename.endswith(".wav"):
            user_path = os.path.join(AUTHORIZED_FOLDER, filename)

            score, prediction = verification.verify_files(
                user_path,
                temp_audio_path
            )

            if bool(prediction) and score > THRESHOLD:
                return filename.replace(".wav", "")

    return None


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

        user = find_authorized_user("temp.wav")
        if user:
            return jsonify({
                "resultado": "valido",
                "usuario": user
                })
        else:
            return jsonify({"resultado": "invalido"})
        
    if not os.path.exists(AUTHORIZED_FOLDER):
        os.makedirs(AUTHORIZED_FOLDER)
        
    return app