import os
import tempfile
import shutil
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import google.generativeai as genai
from faster_whisper import WhisperModel
from gtts import gTTS

# ------------------- Setup -------------------
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (important for Flutter)
user_memory = {}  # In-memory storage per user session

# Load Whisper model once
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

# Setup Google Gemini API
genai.configure(api_key="AIzaSyDidcBB7hjd13yh8VyICMBDFX_wf3HA64A")


# ------------------- Utilities -------------------


# Transcribe Tamil audio using Whisper
def transcribe_audio(input_file):
    segments, _ = whisper_model.transcribe(input_file, language="ta", beam_size=1)
    return " ".join(segment.text for segment in segments)


# Generate AI response using Gemini
def generate_response(prompt):
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(
        prompt, generation_config={"max_output_tokens": 256}
    )
    raw_text = response.text.strip()

    # Remove leading "AI:" or similar prefixes
    if raw_text.lower().startswith("ai:"):
        raw_text = raw_text[3:].strip()

    return raw_text


# Convert Tamil text to speech (MP3)
def text_to_speech(text, temp_dir):
    output_mp3_path = os.path.join(temp_dir, "output.mp3")
    tts = gTTS(text=text, lang="ta")
    tts.save(output_mp3_path)
    return output_mp3_path


# ------------------- Routes -------------------


@app.route("/start_call", methods=["POST"])
def start_call():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    print(f"[START CALL] for user_id: {user_id}")
    user_memory[user_id] = ""  # Initialize memory
    return jsonify({"message": "Call started"}), 200


@app.route("/end_call", methods=["POST"])
def end_call():
    data = request.get_json()
    user_id = data.get("user_id")

    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    print(f"[END CALL] for user_id: {user_id}")
    user_memory.pop(user_id, None)
    return jsonify({"message": "Call ended and memory cleared"}), 200


@app.route("/process_audio", methods=["POST"])
def process_audio():
    if "file" not in request.files or "user_id" not in request.form:
        return jsonify({"error": "Missing file or user_id"}), 400

    user_id = request.form["user_id"]
    file = request.files["file"]

    if user_id not in user_memory:
        return jsonify({"error": "Call not started for user"}), 400

    with tempfile.TemporaryDirectory() as temp_dir:
        input_path = os.path.join(temp_dir, "input.wav")
        file.save(input_path)

        print(f"[TRANSCRIBE] user: {user_id}")
        user_text = transcribe_audio(input_path)

        # Append conversation history
        full_prompt = user_memory[user_id] + "\nUser: " + user_text
        full_prompt += (
            "\nYour name is CallMate AI, created by Vishnu Siva. "
            "Only reply in Tamil. No emojis, symbols, or special characters. "
            "Be concise, natural, and human-like."
        )

        print(f"[PROMPT]: {full_prompt}")
        ai_response = generate_response(full_prompt)
        print(f"[RESPONSE]: {ai_response}")

        # Update memory
        user_memory[user_id] = full_prompt + "\nAI: " + ai_response

        # Convert to MP3 speech
        mp3_path = text_to_speech(ai_response, temp_dir)
        final_path = os.path.join("uploads", f"{user_id}_response.mp3")
        os.makedirs("uploads", exist_ok=True)
        shutil.copy(mp3_path, final_path)

    return send_file(final_path, mimetype="audio/mp3")


# ------------------- Main -------------------
if __name__ == "__main__":
    # Use this for local dev
    app.run(host="0.0.0.0", port=5000, debug=True)
    # For production: use waitress or gunicorn
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8000)
