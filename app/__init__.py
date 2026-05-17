import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from flask import Flask, request, jsonify
from speechbrain.pretrained import SpeakerRecognition
import requests
import subprocess

import torch

user_embeddings = {}

BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
AUTHORIZED_FOLDER = os.path.join(BASE_DIR, "authorized_users")
THRESHOLD = 0.4

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="tmp_model",
    run_opts={"device": "cpu"}
)


def load_authorized_embeddings():
    global user_embeddings
    user_embeddings = {}

    for user_name in os.listdir(AUTHORIZED_FOLDER):
        user_folder = os.path.join(AUTHORIZED_FOLDER, user_name)

        if os.path.isdir(user_folder):

            embeddings_list = []

            for filename in os.listdir(user_folder):
                if filename.endswith(".wav"):
                    path = os.path.join(user_folder, filename)

                    signal = verification.load_audio(path)
                    embedding = verification.encode_batch(signal).squeeze()
                    embeddings_list.append(embedding)

            if embeddings_list:
                # Promedio de embeddings del usuario
                user_embeddings[user_name] = torch.mean(
                    torch.stack(embeddings_list),
                    dim=0
                )

    print("Embeddings cargados correctamente")


def find_authorized_user(temp_audio_path):

    with torch.no_grad():

        signal = verification.load_audio(temp_audio_path)
        test_embedding = verification.encode_batch(signal).squeeze()

        best_score = -1.0
        best_user = None

        for user_name, stored_embedding in user_embeddings.items():

            score = torch.nn.functional.cosine_similarity(
                test_embedding,
                stored_embedding,
                dim=0
            ).item()

            print("Usuario:", user_name)
            print("Score:", score)

            if score > best_score:
                best_score = score
                best_user = user_name

        print("Mejor score:", best_score)

        if best_score >= THRESHOLD:
            return best_user

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

    load_authorized_embeddings()

    return app