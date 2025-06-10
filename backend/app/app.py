import logging
import os
import pickle
import re
from collections import Counter

import google.generativeai as genai
import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.datasets import imdb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# --- IMPROVED CORS Configuration ---
# More explicit CORS setup that works better with Hugging Face Spaces
CORS(app)

# Additional manual CORS headers for better compatibility
@app.after_request
def after_request(response):
    origin = request.headers.get('Origin')
    if origin in [
        'http://localhost:5173',
        'http://localhost:3000',
        'https://nueral-network-frontend.vercel.app'
    ] or (origin and origin.endswith('.vercel.app')):
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# --- Model Loading ---
weights = None
biases = None
idf_values = None
num_words_vocab = 15000

try:
    improved_model_file = 'sentiment_model_v2.pkl'
    if os.path.exists(improved_model_file):
        with open(improved_model_file, 'rb') as f:
            weights, biases, idf_values = pickle.load(f)
        logging.info("Loaded improved model with IDF values (v2)")
    else:
        enhanced_model_file = 'sentiment_model_improved.pkl'
        if os.path.exists(enhanced_model_file):
            with open(enhanced_model_file, 'rb') as f:
                weights, biases, idf_values = pickle.load(f)
            logging.info("Loaded enhanced model with IDF values (v1)")
        else:
            with open('enhanced_model_params.pkl', 'rb') as f:
                weights, biases = pickle.load(f)
            logging.warning("Loading old model format (no IDF values in pickle). Generating IDF values from IMDb dataset.")
            (X_train_sequences, _), _ = imdb.load_data(num_words=num_words_vocab)
            idf_values = generate_idf_values(X_train_sequences, num_words_vocab)
        
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}. Please ensure model files are present.")
    raise FileNotFoundError("Model file is missing")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Load word index for vectorization
word_index = None
word_index_file = 'imdb_word_index.pkl'
if os.path.exists(word_index_file):
    with open(word_index_file, 'rb') as f:
        word_index = pickle.load(f)
    logging.info("Loaded IMDb word index from file.")
else:
    logging.info("IMDb word index file not found. Downloading and saving...")
    try:
        word_index = imdb.get_word_index()
        with open(word_index_file, 'wb') as f:
            pickle.dump(word_index, f)
        logging.info("IMDb word index downloaded and saved.")
    except Exception as e:
        logging.error(f"Failed to download IMDb word index: {e}")
        raise

# --- API Keys and External Service URLs ---
GEMINI_API_URL = 'https://abdul29.pythonanywhere.com/fetch_movie_data'

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

# --- Helper Functions ---
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.\!\?\,\-\']', ' ', text)
    text = text.strip()
    return text

def generate_idf_values(sequences, vocab_size):
    logging.info("Generating IDF values from training data...")
    doc_freqs = np.zeros(vocab_size, dtype=np.int32)
    total_docs = len(sequences)
    
    for sequence in sequences:
        unique_words = set(word_id for word_id in sequence if isinstance(word_id, int) and word_id < vocab_size)
        for word_id in unique_words:
            doc_freqs[word_id] += 1
    
    idf_values = np.log(total_docs / (doc_freqs + 1)) + 1
    return idf_values

def vectorize_single_review(review, word_index_dict, idf_vals, vocab_size=15000):
    review = preprocess_text(review)
    words = review.split()
    
    indices = []
    for word in words:
        if word in word_index_dict:
            idx = word_index_dict[word] + 3
            if 3 <= idx < vocab_size:
                indices.append(idx)
    
    if not indices:
        logging.warning("No valid words found in review for vectorization.")
        return np.zeros((1, vocab_size), dtype=np.float32)
    
    word_counts = Counter(indices)
    result = np.zeros((vocab_size,), dtype=np.float32)
    
    for idx, freq in word_counts.items():
        if idx < len(idf_vals):
            tf = 1 + np.log(freq)
            result[idx] = tf * idf_vals[idx]
    
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    
    return result.reshape(1, -1)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(np.clip(x_shifted, -500, 500))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-8)

def forward_pass(x, weights, biases, training=False):
    activations = [x]
    
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = leaky_relu(z)
        
        activations.append(a)
    
    return activations[-1]

def predict_probabilities(x, weights, biases):
    return forward_pass(x, weights, biases, training=False)

def interpret_sentiment_with_passage(review_vector, weights, biases):
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = predict_probabilities(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]
    confidence = abs(positive_prob - 0.5) * 2

    logging.info(f"Local Model: Positive Probability: {positive_prob:.4f}, Confidence: {confidence:.4f}")

    if positive_prob >= 0.9:
        sentiment_classification = "Overwhelmingly Positive"
        descriptive_passage = "This movie delivers an exceptional cinematic experience with outstanding performances, compelling storytelling, and masterful direction."
    elif positive_prob >= 0.8:
        sentiment_classification = "Highly Positive" 
        descriptive_passage = "A genuinely impressive film that excels in multiple aspects, offering strong entertainment value and memorable moments."
    elif positive_prob >= 0.65:
        sentiment_classification = "Positive"
        descriptive_passage = "This film succeeds in delivering an engaging and well-crafted experience with solid performances and effective storytelling."
    elif positive_prob >= 0.55:
        sentiment_classification = "Moderately Positive"
        descriptive_passage = "While not perfect, this film offers more strengths than weaknesses with decent execution and genuine quality moments."
    elif positive_prob >= 0.45:
        sentiment_classification = "Neutral/Mixed"
        descriptive_passage = "This film presents a balanced mix of positive and negative elements, resulting in an average viewing experience."
    elif positive_prob >= 0.35:
        sentiment_classification = "Moderately Negative"
        descriptive_passage = "Despite some redeeming qualities, this film suffers from notable issues that detract from the overall experience."
    elif positive_prob >= 0.2:
        sentiment_classification = "Negative"
        descriptive_passage = "This film fails to deliver on multiple fronts with significant problems that make it difficult to recommend."
    elif positive_prob >= 0.1:
        sentiment_classification = "Highly Negative"
        descriptive_passage = "A largely unsuccessful film with major flaws that severely impact the viewing experience."
    else:
        sentiment_classification = "Overwhelmingly Negative"
        descriptive_passage = "This film represents a significant misfire with fundamental problems across all aspects of production."

    return sentiment_classification, descriptive_passage, positive_prob, confidence

# --- Flask Routes ---
@app.route('/health')
def health():
    return "OK", 200

@app.route('/')
def home():
    return "Improved Movie Sentiment Analysis API is live!"

# EXPLICIT OPTIONS HANDLER
@app.route('/predict', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    origin = request.headers.get('Origin')
    if origin in [
        'http://localhost:5173',
        'http://localhost:3000',
        'https://nueral-network-frontend.vercel.app'
    ] or (origin and origin.endswith('.vercel.app')):
        response.headers.add('Access-Control-Allow-Origin', origin)
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept,Origin,X-Requested-With')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Received request for local sentiment analysis or movie review fetch.")
    try:
        data = request.get_json()
        user_input = data.get('text')

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        movie_details = None
        reviews = []
        source_of_analysis = "User Provided Reviews"

        if user_input.startswith('#'):
            movie_name = user_input[1:].strip()
            if not movie_name:
                return jsonify({'error': 'Movie name not provided after #'}), 400

            logging.info(f"Fetching reviews for movie: {movie_name} from Gemini API service at {GEMINI_API_URL}")
            try:
                response = requests.post(GEMINI_API_URL, json={'query': movie_name}, timeout=20)

                if response.status_code != 200:
                    logging.error(f"Gemini API service responded with status {response.status_code}: {response.text}")
                    return jsonify({'error': f'Failed to fetch movie data: {response.text}'}), response.status_code
                
                response.raise_for_status()
                gemini_data = response.json()
                movie_details = gemini_data.get('details')
                reviews = gemini_data.get('reviews')
                source_of_analysis = "Online Reviews (via Gemini API)"

                if not reviews or len(reviews) == 0:
                    reviews = ["No specific reviews found by Gemini for this movie. Analyzing generic sentiment."]
                    logging.warning(f"No reviews returned by Gemini for '{movie_name}'. Proceeding with default review.")
                    
            except requests.exceptions.Timeout:
                logging.error("Timeout when connecting to Gemini API service.")
                return jsonify({'error': 'Gemini API service timeout. Please try again.'}), 504
            except requests.exceptions.ConnectionError as e:
                logging.error(f"Cannot connect to Gemini API service: {e}")
                return jsonify({'error': 'Gemini Movie Data service is unreachable. Please ensure your Gemini proxy service is running.'}), 503
            except requests.exceptions.RequestException as e:
                logging.error(f"Request to Gemini API service failed: {e}")
                return jsonify({'error': f'Failed to communicate with Gemini API service: {str(e)}'}), 500
                
        else:
            reviews = [user_input]
            movie_details = {"title": "User Provided Review", "year": "N/A", "genre": "N/A"}
            source_of_analysis = "Direct User Input"

        if not isinstance(reviews, list):
            reviews = [str(reviews)]
        reviews_for_model = [str(r) for r in reviews if r and str(r).strip()]

        if not reviews_for_model:
            return jsonify({'error': 'No valid reviews to analyze after processing input.'}), 400

        logging.info(f"Analyzing sentiment of the first review: {reviews_for_model[0][:100]}...")
        
        first_review_vector = vectorize_single_review(
            reviews_for_model[0], 
            word_index, 
            idf_values, 
            num_words_vocab
        )
        
        logging.info(f"Review vector shape: {first_review_vector.shape}")

        sentiment_classification, descriptive_passage, positive_prob, confidence = interpret_sentiment_with_passage(
            first_review_vector, weights, biases
        )
        
        return jsonify({
            'sentiment_classification': sentiment_classification,
            'descriptive_passage': descriptive_passage,
            'source': source_of_analysis,
            'details': movie_details,
            'reviews': reviews,
            'positive_probability': float(positive_prob),
            'confidence': float(confidence),
            'model_version': 'Improved v3 (Local)',
            'vocabulary_size': num_words_vocab
        })

    except Exception as e:
        logging.error(f"Unhandled error in /predict endpoint: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error in local prediction: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def proxy_huggingface_predict():
    logging.info("Received request for Hugging Face proxy.")
    
    if not request.is_json:
        logging.error("Proxy request: Request must be JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_to_process = data.get("text")

    if not text_to_process:
        logging.error("Proxy request: Missing 'text' in request body")
        return jsonify({"error": "Missing 'text' in request body"}), 400

    try:
        logging.info(f"Proxying request to Hugging Face API: {HUGGING_FACE_PREDICT_URL}")
        hf_response = requests.post(
            HUGGING_FACE_PREDICT_URL,
            headers={"Content-Type": "application/json"},
            json={"text": text_to_process},
            timeout=30
        )
        hf_response.raise_for_status()

        logging.info("Successfully proxied request to Hugging Face.")
        return jsonify(hf_response.json()), hf_response.status_code

    except requests.exceptions.Timeout:
        logging.error("Proxy request: Timeout when connecting to Hugging Face API.")
        return jsonify({"error": "Hugging Face API timeout. Please try again."}), 504
    except requests.exceptions.ConnectionError as e:
        logging.error(f"Proxy request: Cannot connect to Hugging Face API: {e}")
        return jsonify({"error": "Hugging Face API is unreachable. Please check the service status."}), 503
    except requests.exceptions.RequestException as e:
        logging.error(f"Proxy request: Request to Hugging Face API failed: {e}", exc_info=True)
        return jsonify({"error": f"Failed to connect to Hugging Face API: {str(e)}", "details": hf_response.text if 'hf_response' in locals() else "No response body"}), 500
    except ValueError:
        logging.error("Proxy request: Invalid JSON response from Hugging Face API.")
        return jsonify({"error": "Invalid JSON response from Hugging Face API"}), 502
    except Exception as e:
        logging.error(f"Proxy request: Unhandled error in proxy endpoint: {e}", exc_info=True)
        return jsonify({'error': f'Internal server error in proxy: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.getenv("APP_PORT", 5000))
    logging.info(f"Starting Improved Movie Sentiment Analysis API on port {port}")
    logging.info(f"Model vocabulary size: {num_words_vocab}")
    logging.info(f"Using IDF values: {len(idf_values) if idf_values is not None else 'None'}")
    app.run(host="0.0.0.0", port=port, debug=False)