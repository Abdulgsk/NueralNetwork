# app.py
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import model  # your model.py with predict_from_reviews and load_model functions
import requests  # For calling the Gemini API service
from dotenv import load_dotenv
import os
import pickle
from tensorflow.keras.datasets import imdb
import numpy as np
from scipy.sparse import csr_matrix

# Load the .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load your custom sentiment model only once at startup
# Load model parameters once on startup
try:
    # Updated filename to match your model.py
    with open('enhanced_model_params.pkl', 'rb') as f:
        weights, biases = pickle.load(f)
except FileNotFoundError:
    logging.error("Model file 'enhanced_model_params.pkl' not found")
    raise FileNotFoundError("Model file is missing")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Updated vocabulary size to match your model.py
num_words_vocab = 10000

import pickle
word_index_file = 'imdb_word_index.pkl'
if os.path.exists(word_index_file):
    with open(word_index_file, 'rb') as f:
        word_index = pickle.load(f)
else:
    word_index = imdb.get_word_index()
    with open(word_index_file, 'wb') as f:
        pickle.dump(word_index, f)

# This URL now points to your separate Gemini Flask API service
GEMINI_API_URL = 'https://nueralnetwork-production.up.railway.app/fetch_movie_data'

# Configure Gemini API key for general-purpose text generation in this app
# Make sure to replace 'YOUR_GEMINI_API_KEY_HERE' with your actual API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")
genai.configure(api_key=api_key)

@app.route('/health')
def health():
    return "OK", 200

@app.route('/')
def home():
    return "Movie Sentiment Analysis API is live!"

@app.route('/generate-reason', methods=['POST'])
def generate_reason():
    try:
        data = request.get_json()
        short_reason = data.get('shortReason')

        if not short_reason:
            return jsonify({'error': 'shortReason not provided'}), 400

        llm_model = genai.GenerativeModel('gemini-1.5-flash')
        chat = llm_model.start_chat()
        prompt = f'Expand this into a formal and detailed leave request reason:\n\n"{short_reason}"'
        response = chat.send_message(prompt)

        return jsonify({'fullReason': response.text})

    except Exception as e:
        logging.error(f"Error generating reason: {e}")
        return jsonify({'error': 'Failed to generate detailed reason'}), 500


def vectorize_single_review(review):
    """
    Updated vectorization function to match the enhanced model.py approach
    """
    words = review.lower().split()
    indices = [word_index.get(word, 0) + 3 for word in words if word in word_index]
    indices = [i for i in indices if i < num_words_vocab and i >= 3]
    
    # Create weighted vector (matching model.py approach)
    if not indices:
        return np.zeros((num_words_vocab,), dtype=np.float32)
    
    word_freq = {}
    for idx in indices:
        word_freq[idx] = word_freq.get(idx, 0) + 1
    
    result = np.zeros((num_words_vocab,), dtype=np.float32)
    for idx, freq in word_freq.items():
        result[idx] = np.log(freq + 1)  # TF weighting
    
    # Normalize
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    
    return result.reshape(1, -1)

# Updated activation functions to match model.py
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def forward_pass(x, weights, biases):
    """
    Updated forward pass to match the enhanced model.py approach
    """
    activations = [x]
    
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if i == len(weights) - 1:  # Output layer
            a = softmax(z)
        else:  # Hidden layers
            a = leaky_relu(z)  # Using Leaky ReLU to match model.py
        
        activations.append(a)
    
    return activations[-1]  # Return final output

def interpret_sentiment_with_passage(review_vector, weights, biases):
    """
    Updated sentiment interpretation function to match model.py
    """
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = forward_pass(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]
    confidence = max(positive_prob, 1 - positive_prob)  # Confidence in the prediction

    if positive_prob >= 0.95:
        sentiment_classification = "Overwhelmingly Positive"
        descriptive_passage = "This movie delivers an exceptional cinematic experience with outstanding performances, compelling storytelling, and masterful direction that resonates deeply with audiences."
    elif positive_prob >= 0.85:
        sentiment_classification = "Highly Positive"
        descriptive_passage = "A genuinely impressive film that excels in multiple aspects, offering strong entertainment value and memorable moments that justify enthusiastic recommendation."
    elif positive_prob >= 0.70:
        sentiment_classification = "Positive"
        descriptive_passage = "This film succeeds in delivering an engaging and well-crafted experience with solid performances and effective storytelling that entertains throughout."
    elif positive_prob >= 0.55:
        sentiment_classification = "Moderately Positive"
        descriptive_passage = "While not without minor flaws, this film offers more strengths than weaknesses with decent execution and moments of genuine quality."
    elif positive_prob >= 0.45:
        sentiment_classification = "Neutral/Mixed"
        descriptive_passage = "This film presents a balanced mix of positive and negative elements, resulting in an average viewing experience with both highlights and disappointments."
    elif positive_prob >= 0.30:
        sentiment_classification = "Moderately Negative"
        descriptive_passage = "Despite some redeeming qualities, this film suffers from notable issues in execution, pacing, or storytelling that detract from the overall experience."
    elif positive_prob >= 0.15:
        sentiment_classification = "Negative"
        descriptive_passage = "This film fails to deliver on multiple fronts with significant problems in direction, performance, or script that make it difficult to recommend."
    elif positive_prob >= 0.05:
        sentiment_classification = "Highly Negative"
        descriptive_passage = "A largely unsuccessful film with major flaws that severely impact the viewing experience, offering little in terms of entertainment or artistic value."
    else:
        sentiment_classification = "Overwhelmingly Negative"
        descriptive_passage = "This film represents a significant misfire with fundamental problems across all aspects of production, resulting in a thoroughly disappointing experience."

    return sentiment_classification, descriptive_passage, positive_prob, confidence

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('text')

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        movie_details = None
        reviews = []
        source_of_analysis = "User Provided Review"  # Default source

        # Check if input is a movie name (e.g., #TheMatrix)
        if user_input.startswith('#'):
            movie_name = user_input[1:].strip()
            if not movie_name:
                return jsonify({'error': 'Movie name not provided after #'}), 400

            logging.info(f"Fetching reviews for movie: {movie_name} from Gemini API service at {GEMINI_API_URL}")
            # Fetch reviews and details from the separate Gemini API service
            response = requests.post(GEMINI_API_URL, json={'query': movie_name}, timeout=10)

            if response.status_code != 200:
                logging.error(f"Error from Gemini API service: {response.status_code} - {response.text}")
                return jsonify({'error': f'Failed to fetch from Gemini API service: {response.text}'}), response.status_code

            gemini_data = response.json()
            movie_details = gemini_data.get('details')
            reviews = gemini_data.get('reviews')
            source_of_analysis = "Online Reviews via Gemini"

            if not reviews or len(reviews) == 0:
                reviews = ["No specific reviews found by Gemini for this movie."]
                logging.error(f"No reviews returned by Gemini for '{movie_name}'. Proceeding with default review.")

        else:
            # Use direct user review
            reviews = [user_input]
            movie_details = {"title": "User Provided Review", "year": "N/A", "genre": "N/A"}
            source_of_analysis = "Direct User Input"

        # Ensure reviews are always a list of strings
        if not isinstance(reviews, list):
            reviews = [str(reviews)]
        reviews_for_model = [str(r) for r in reviews if r]

        if not reviews_for_model:
            return jsonify({'error': 'No valid reviews to analyze after processing input.'}), 400

        # Predict sentiment for the first review using updated functions
        logging.info(f"Vectorizing review: {reviews_for_model[0][:100]}...")
        first_review_vector = vectorize_single_review(reviews_for_model[0])
        logging.info(f"Review vector shape: {first_review_vector.shape}")
        
        # Use the updated interpretation function
        sentiment_classification, descriptive_passage, positive_prob, confidence = interpret_sentiment_with_passage(
            first_review_vector, weights, biases
        )

        return jsonify({
            'sentiment_classification': sentiment_classification,
            'descriptive_passage': descriptive_passage,
            'source': source_of_analysis,
            'details': movie_details,
            'reviews': reviews,
            'positive_probability': float(positive_prob),  # Convert to float for JSON serialization
            'confidence': float(confidence)  # Added confidence score
        })

    except requests.exceptions.ConnectionError:
        return jsonify({'error': 'Gemini Movie Data service is unreachable. Please ensure gemini_movie_api.py is running on port 5001.'}), 503
    except Exception as e:
        logging.error(f"Unhandled error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.getenv("APP_PORT", 5000))
    app.run(host="0.0.0.0", port=port)