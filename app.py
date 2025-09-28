import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Hugging Face থেকে পাওয়া আপনার Access Token এখানে বসাতে হবে
# তবে আমরা এটি সরাসরি কোডে না বসিয়ে Environment Variable হিসেবে ব্যবহার করব
API_TOKEN = os.environ.get("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
headers = {"Authorization": f"Bearer {API_TOKEN}"}

@app.route('/')
def home():
    return "Backend server is running!"

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Hugging Face API-তে ফাইল পাঠানো হচ্ছে
        response = requests.post(API_URL, headers=headers, data=file.read())
        result = response.json()

        if 'error' in result:
            # যদি মডেল লোড হতে সময় লাগে, Hugging Face একটি এরর দেয়
            if 'is currently loading' in result.get('error', ''):
                estimated_time = result.get('estimated_time', 20)
                return jsonify({'error': f'Model is loading, please try again in {int(estimated_time)} seconds.'}), 503
            return jsonify({'error': result.get('error')}), 500
        
        return jsonify({
            'success': True,
            'fileName': file.filename,
            'transcript': result.get('text')
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
