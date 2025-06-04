import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import os
import pickle
from tensorflow.keras.datasets import imdb
import numpy as np
from collections import Counter
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

# Updated model loading to handle the improved model format
try:
    # Try loading the improved model format first
    improved_model_file = 'sentiment_model_improved.pkl'
    if os.path.exists(improved_model_file):
        with open(improved_model_file, 'rb') as f:
            weights, biases, idf_values = pickle.load(f)
        logging.info("Loaded improved model with IDF values")
        num_words_vocab = 15000  # Updated to match the improved model
    else:
        # Fallback to enhanced model format
        enhanced_model_file = 'sentiment_model_v2.pkl'
        if os.path.exists(enhanced_model_file):
            with open(enhanced_model_file, 'rb') as f:
                weights, biases, idf_values = pickle.load(f)
            logging.info("Loaded enhanced model with IDF values")
            num_words_vocab = 15000
        else:
            # Fallback to old model format
            with open('enhanced_model_params.pkl', 'rb') as f:
                weights, biases = pickle.load(f)
            # Generate IDF values if not available
            logging.warning("Loading old model format, generating IDF values...")
            (X_train_sequences, _), _ = imdb.load_data(num_words=15000)
            idf_values = generate_idf_values(X_train_sequences, 15000)
            num_words_vocab = 15000
        
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}")
    raise FileNotFoundError("Model file is missing")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise

# Load word index for vectorization
word_index_file = 'imdb_word_index.pkl'
if os.path.exists(word_index_file):
    with open(word_index_file, 'rb') as f:
        word_index = pickle.load(f)
else:
    word_index = imdb.get_word_index()
    with open(word_index_file, 'wb') as f:
        pickle.dump(word_index, f)

GEMINI_API_URL = 'https://nueralnetwork-production.up.railway.app/fetch_movie_data'

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY not found in .env file")
    raise ValueError("GEMINI_API_KEY is required")
genai.configure(api_key=api_key)

def preprocess_text(text):
    """Enhanced text preprocessing - matches the improved model"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\-\']', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def generate_idf_values(sequences, vocab_size):
    """Generate IDF values from training sequences (fallback for old models)"""
    logging.info("Generating IDF values from training data...")
    doc_freqs = np.zeros(vocab_size, dtype=np.int32)
    total_docs = len(sequences)
    
    for sequence in sequences:
        unique_words = set(word_id for word_id in sequence if word_id < vocab_size)
        for word_id in unique_words:
            doc_freqs[word_id] += 1
    
    # Updated IDF calculation to match improved model
    idf_values = np.log(total_docs / (doc_freqs + 1)) + 1  # Add 1 to prevent negative values
    return idf_values

def vectorize_single_review(review, word_index_dict, idf_vals, vocab_size=15000):
    """
    Improved vectorization function that matches the training approach
    Uses TF-IDF with L2 normalization and improved preprocessing
    """
    # Preprocess the review text (matches improved model)
    review = preprocess_text(review)
    words = review.split()
    
    # Convert words to indices with proper offset (IMDb uses +3 offset)
    indices = []
    for word in words:
        if word in word_index_dict:
            idx = word_index_dict[word] + 3  # IMDb uses +3 offset
            if 3 <= idx < vocab_size:  # Valid range
                indices.append(idx)
    
    if not indices:
        logging.warning("No valid words found in review")
        return np.zeros((vocab_size,), dtype=np.float32)
    
    # Calculate term frequencies
    word_counts = Counter(indices)
    result = np.zeros((vocab_size,), dtype=np.float32)
    
    # Apply improved TF-IDF transformation (matches improved model)
    for idx, freq in word_counts.items():
        if idx < len(idf_vals):
            # Improved TF calculation - prevents zero values
            tf = 1 + np.log(freq)  # This matches the improved model
            result[idx] = tf * idf_vals[idx]
    
    # L2 normalization
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    
    return result.reshape(1, -1)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    # More stable softmax implementation (matches improved model)
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(np.clip(x_shifted, -500, 500))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-8)

def forward_pass(x, weights, biases, training=False):
    """Forward pass through the neural network - matches improved model"""
    activations = [x]
    
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if i == len(weights) - 1:  # Output layer
            a = softmax(z)
        else:  # Hidden layers
            a = leaky_relu(z)
        
        activations.append(a)
    
    return activations[-1]

def predict_probabilities(x, weights, biases):
    """Get prediction probabilities"""
    return forward_pass(x, weights, biases, training=False)

def interpret_sentiment_with_passage(review_vector, weights, biases):
    """
    Improved sentiment interpretation with more balanced thresholds
    Matches the improved model's classification system
    """
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = predict_probabilities(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]
    confidence = abs(positive_prob - 0.5) * 2  # Confidence based on distance from neutral

    logging.info(f"Positive Probability: {positive_prob:.4f}, Confidence: {confidence:.4f}")

    # More balanced thresholds (matches improved model)
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

@app.route('/health')
def health():
    return "OK", 200

@app.route('/')
def home():
    return "Improved Movie Sentiment Analysis API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        user_input = data.get('text')

        if not user_input:
            return jsonify({'error': 'No input provided'}), 400

        movie_details = None
        reviews = []
        source_of_analysis = "User Provided Reviews"

        # Handle movie search (input starting with #)
        if user_input.startswith('#'):
            movie_name = user_input[1:].strip()
            if not movie_name:
                return jsonify({'error': 'Movie name not provided after #'}), 400

            logging.info(f"Fetching reviews for movie: {movie_name} from Gemini API service at {GEMINI_API_URL}")
            try:
                response = requests.post(GEMINI_API_URL, json={'query': movie_name}, timeout=15)

                if response.status_code != 200:
                    logging.error(f"{response.status_code} - {response.text}")
                    return jsonify({'error': f' {response.text}'}), response.status_code

                gemini_data = response.json()
                movie_details = gemini_data.get('details')
                reviews = gemini_data.get('reviews')
                source_of_analysis = "Online Reviews"

                if not reviews or len(reviews) == 0:
                    reviews = ["No specific reviews found by Gemini for this movie."]
                    logging.warning(f"No reviews returned by Gemini for '{movie_name}'. Proceeding with default review.")
                    
            except requests.exceptions.Timeout:
                logging.error("Timeout when connecting to Gemini API service")
                return jsonify({'error': 'Gemini API service timeout. Please try again.'}), 504
            except requests.exceptions.ConnectionError:
                logging.error("Cannot connect to Gemini API service")
                return jsonify({'error': 'Gemini Movie Data service is unreachable. Please ensure gemini_movie_api.py is running on port 5001.'}), 503
                
        else:
            # Direct review analysis
            reviews = [user_input]
            movie_details = {"title": "User Provided Review", "year": "N/A", "genre": "N/A"}
            source_of_analysis = "Direct User Input"

        # Ensure reviews is a list of strings
        if not isinstance(reviews, list):
            reviews = [str(reviews)]
        reviews_for_model = [str(r) for r in reviews if r and str(r).strip()]

        if not reviews_for_model:
            return jsonify({'error': 'No valid reviews to analyze after processing input.'}), 400

        logging.info(f"Analyzing review: {reviews_for_model[0][:100]}...")
        
        # Use the improved vectorization function
        first_review_vector = vectorize_single_review(
            reviews_for_model[0], 
            word_index, 
            idf_values, 
            num_words_vocab
        )
        
        logging.info(f"Review vector shape: {first_review_vector.shape}")
        logging.info(f"Review vector stats: mean={first_review_vector.mean():.6f}, std={first_review_vector.std():.6f}")

        # Get sentiment analysis
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
            'model_version': 'Improved v3',
            'vocabulary_size': num_words_vocab
        })

    except Exception as e:
        logging.error(f"Unhandled error in predict endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.getenv("APP_PORT", 5000))
    logging.info(f"Starting Improved Movie Sentiment Analysis API on port {port}")
    logging.info(f"Model vocabulary size: {num_words_vocab}")
    logging.info(f"Using IDF values: {len(idf_values) if idf_values is not None else 'None'}")
    app.run(host="0.0.0.0", port=port, debug=False)