import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# প্রোডাকশন-গ্রেড লগিং সেটআপ
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Render-এ সেট করা Hugging Face টোকেন লোড করা
API_TOKEN = os.environ.get("HF_API_TOKEN")

# আলট্রা-ফাস্ট এবং নির্ভুল মডেল: Distil-Whisper Large v2
API_URL = "https://api-inference.huggingface.co/models/distil-whisper/distil-large-v2"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

# অনুমোদিত ফাইলের ধরন (অডিও এবং ভিডিও)
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'flac', 'mp4', 'mov', 'avi', 'mkv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    """সার্ভারটি চলছে কিনা তা পরীক্ষা করার জন্য হোম রুট।"""
    return "Your TranscribeAI backend is live and running flawlessly!"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """অডিও/ভিডিও ফাইল গ্রহণ করে এবং Hugging Face API ব্যবহার করে ট্রান্সক্রাইব করে।"""
    
    app.logger.info("Received a new transcription request.")

    if 'file' not in request.files:
        app.logger.error("Request Error: No file part in the request.")
        return jsonify({'error': 'No file part found. Please upload a file.'}), 400

    file = request.files['file']

    if file.filename == '':
        app.logger.error("Request Error: No file was selected.")
        return jsonify({'error': 'No file selected for upload.'}), 400

    # ফাইলটি অনুমোদিত ধরনের কিনা তা পরীক্ষা করা
    if not allowed_file(file.filename):
        app.logger.error(f"Invalid file type: {file.filename}")
        return jsonify({'error': 'Invalid file type. Please upload an audio or video file.'}), 400

    app.logger.info(f"Processing file: {file.filename}")

    try:
        # Hugging Face API-তে ফাইল ডেটা পাঠানো
        response = requests.post(API_URL, headers=headers, data=file.read())
        
        result = response.json()
        app.logger.info(f"Response from Hugging Face: {result}")

        if 'error' in result:
            error_message = result.get('error')
            app.logger.error(f"Hugging Face API Error: {error_message}")

            if 'is currently loading' in error_message:
                estimated_time = result.get('estimated_time', 20)
                return jsonify({'error': f'AI model is starting up. Please try again in {int(estimated_time)} seconds.'}), 503
            
            return jsonify({'error': f'AI Service Error: {error_message}'}), 500
        
        transcript_text = result.get('text', 'Could not retrieve transcript text.')
        app.logger.info("Transcription successful.")
        
        return jsonify({
            'success': True,
            'fileName': file.filename,
            'transcript': transcript_text.strip()
        })

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error connecting to Hugging Face: {e}")
        return jsonify({'error': 'Could not connect to the AI service. Please check your network and try again.'}), 504
    except Exception as e:
        app.logger.error(f"An unexpected server error occurred: {e}", exc_info=True)
        return jsonify({'error': 'An internal server error occurred. The issue has been logged.'}), 500
