import os
import time

from flask import Flask, jsonify, render_template, request

from binge_buddy.conversational_agent_manager import ConversationalAgentManager
from binge_buddy.memory_db import MemoryDB
from binge_buddy.memory_handler import EpisodicMemoryHandler, SemanticMemoryHandler
from binge_buddy.message_log import MessageLog
from binge_buddy.ollama import OllamaLLM
from binge_buddy.perception.audito_transcriber import AudioTranscriber
from binge_buddy.perception.sentiment_analyzer import SentimentAnalyzer


class FrontEnd:
    def __init__(self, mode):
        # Set up the Flask app
        self.app = Flask(__name__)
        self.llm = OllamaLLM()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.memory_db = MemoryDB()
        self.mode = mode
        if self.mode == "semantic":
            memory_handler = SemanticMemoryHandler(self.memory_db)
        else:
            memory_handler = EpisodicMemoryHandler(self.memory_db)

        self.message_log = MessageLog(
            user_id="user",
            session_id="session",
            memory_handler=memory_handler,
            mode=mode,
        )
        self.conversational_agent_manager = ConversationalAgentManager(
            self.llm, self.message_log, memory_handler
        )
        self.audio_transcriber = AudioTranscriber()
        self.set_up_routes()

    def set_up_routes(self):
        @self.app.route("/")
        def index():
            return render_template("front_end.html")

        @self.app.route("/send_message", methods=["POST"])
        def handle_user_message():
            data = request.get_json()
            if "text" in data:
                user_message = data["text"]
                print(f"User message received: {user_message}")
                # todo: update userID and sessionID
                message = self.sentiment_analyzer.extract_emotion(
                    user_message, "userId", "sessionId"
                )
                ca = self.conversational_agent_manager.get_agent("userId", "sessionId")
                response = ca.process_message(message)
                return jsonify({"response": response})

        @self.app.route("/upload", methods=["POST"])
        def upload_audio():
            if "audio" not in request.files:
                return jsonify({"error": "No audio file uploaded"}), 400

            audio_file = request.files["audio"]

            os.makedirs("./audio", exist_ok=True)

            filename = f"{int(time.time())}_{audio_file.filename}"
            file_path = os.path.join("./audio", filename)

            # Save the audio file to disk
            audio_file.save(file_path)

            transcribed_text = self.audio_transcriber.transcribe(file_path)

            os.remove(file_path)

            return jsonify({"transcribed_text": transcribed_text})

    def run_flask(self):
        """Start the Flask application."""
        self.app.run(port=6555, host="0.0.0.0", debug=True)


if __name__ == "__main__":
    sample_memory = {
        "user_id": "kanta_001",
        "name": "Kanta",
        "memories": {
            "LIKES": "Kanta likes sci-fi and action movies but he also enjoys rom-coms.",
            "DISLIKES": "Kanta hates Japanese movies, especially anime.",
            "FAVOURITES": "Kanta's favorite movie is Inception, and he loves Christopher Nolan's films.",
            "GENRE": "Kanta prefers movies with deep storytelling, complex characters, and mind-bending plots.",
        },
        "last_updated": "2024-03-09T12:00:00Z",
    }

    front_end = FrontEnd(mode="semantic")
    front_end.run_flask()
