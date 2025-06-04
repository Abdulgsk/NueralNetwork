logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)

import logging
import requests # Make sure requests is imported
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup

# Optimized logging - set to INFO for better debugging during development

load_dotenv()

app = Flask(__name__)

CORS(app)
# --- Corrected CORS Configuration ---
# Remove the redundant CORS(app) line. Keep only the specific one.
# Replace your current CORS config with this in both services:
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://nueral-network-frontend.vercel.app",
            "https://positive-playfulness-production.up.railway.app",
            "https://nueralnetwork-production.up.railway.app"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Removed the global 'session' object

class OptimizedMovieFetcher:
    def __init__(self):
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.omdb_base_url = "http://www.omdbapi.com"
        # Define a common User-Agent header for direct requests
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
    
    def get_movie_details_omdb(self, title=None, imdb_id=None):
        """Optimized OMDB API call with better error handling"""
        if not OMDB_API_KEY:
            logging.warning("OMDB_API_KEY not set. Cannot fetch movie details from OMDB.")
            return None
            
        try:
            params = {'apikey': OMDB_API_KEY, 'plot': 'full'}
            
            if imdb_id:
                params['i'] = imdb_id
            elif title:
                params['t'] = title
            else:
                logging.warning("No title or IMDb ID provided for OMDB lookup.")
                return None
                
            # Use direct requests.get() instead of session.get()
            response = requests.get(self.omdb_base_url, params=params, headers=self.headers, timeout=8)
            response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()
            
            if data.get('Response') != 'True':
                logging.warning(f"OMDB API returned an error for query '{title or imdb_id}': {data.get('Error', 'Unknown OMDB error')}")
                return None
                
            return self._parse_omdb_data(data)
            
        except requests.exceptions.RequestException as req_e:
            logging.error(f"OMDB API request error for '{title or imdb_id}': {req_e}")
            return None
        except Exception as e:
            logging.error(f"OMDB processing error for '{title or imdb_id}': {e}")
            return None
    
    def _parse_omdb_data(self, data):
        """Streamlined data parsing"""
        def safe_int(value):
            if value and value != 'N/A':
                try:
                    return int(''.join(filter(str.isdigit, str(value))))
                except ValueError:
                    pass
            return None
        
        def safe_float(value):
            if value and value != 'N/A':
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
            return None
        
        def split_list(value):
            return [item.strip() for item in value.split(',')] if value and value != 'N/A' else []
        
        # Parse runtime efficiently
        runtime = None
        if data.get('Runtime') and data.get('Runtime') != 'N/A':
            runtime_nums = ''.join(filter(str.isdigit, data.get('Runtime')))
            runtime = int(runtime_nums) if runtime_nums else None
        
        # Parse ratings efficiently
        ratings = []
        if data.get('Ratings'):
            for rating in data.get('Ratings', []):
                ratings.append({
                    'source': rating.get('Source'),
                    'value': rating.get('Value')
                })
        
        return {
            'title': data.get('Title'),
            'year': safe_int(data.get('Year')),
            'rated': data.get('Rated') if data.get('Rated') != 'N/A' else None,
            'released_date': data.get('Released') if data.get('Released') != 'N/A' else None,
            'runtime_minutes': runtime,
            'genres': split_list(data.get('Genre')),
            'director': data.get('Director') if data.get('Director') != 'N/A' else None,
            'actors': split_list(data.get('Actors')),
            'plot': data.get('Plot') if data.get('Plot') != 'N/A' else None,
            'languages': split_list(data.get('Language')),
            'countries': split_list(data.get('Country')),
            'awards': data.get('Awards') if data.get('Awards') != 'N/A' else None,
            'poster_url': data.get('Poster') if data.get('Poster') != 'N/A' else None,
            'ratings': ratings,
            'imdb_id': data.get('imdbID'),
            'imdb_votes': data.get('imdbVotes') if data.get('imdbVotes') != 'N/A' else None,
            'box_office': safe_int(data.get('BoxOffice')),
            'metascore': safe_int(data.get('Metascore')),
            'response': data.get('Response')
        }

def find_movie_name_optimized(movie_query):
    """Optimized movie name correction with faster prompt"""
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set. Skipping movie name correction.")
        return movie_query
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Fix this movie title if misspelled, return exact official title only:
        "{movie_query}"
        
        Examples:
        "avilars" → "Avatar"
        "dark nigt" → "The Dark Knight"
        
        If correct, return as-is. One line response only."""
        
        response = model.generate_content(prompt)
        corrected = response.text.strip()
        
        return corrected if corrected and len(corrected) > 1 else movie_query
        
    except Exception as e:
        logging.error(f"Gemini error during movie name correction: {e}")
        return movie_query

def generate_fast_reviews(movie_title, count=10):
    """Generate reviews with optimized prompt"""
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set. Returning static reviews.")
        return get_static_reviews()
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""Generate {count} short movie reviews for "{movie_title}". Mix of positive/negative. JSON format:
        [{{"review_text": "Great movie!", "rating": "8/10", "author": "User1", "sentiment": "positive"}}]
        Keep reviews 1-2 sentences. No extra text."""
        
        response = model.generate_content(prompt)
        reviews_text = response.text.strip()
        
        # Clean JSON
        if reviews_text.startswith('```json'):
            reviews_text = reviews_text[7:-3]
        reviews_text = reviews_text.strip()
        
        reviews = json.loads(reviews_text)
        return reviews if isinstance(reviews, list) else get_static_reviews()
        
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in generated reviews: {e}. Raw text: {reviews_text}")
        return get_static_reviews()
    except Exception as e:
        logging.error(f"Error generating fast reviews: {e}")
        return get_static_reviews()

def get_static_reviews():
    """Fast fallback reviews"""
    return [
        {"review_text": "Great movie! Really enjoyed it.", "rating": "8/10", "author": "CinemaFan", "sentiment": "positive"},
        {"review_text": "Average film, not bad but not great.", "rating": "6/10", "author": "MovieWatcher", "sentiment": "neutral"},
        {"review_text": "Disappointing, expected better.", "rating": "4/10", "author": "CriticUser", "sentiment": "negative"},
        {"review_text": "Absolutely loved it! Highly recommend.", "rating": "9/10", "author": "FilmLover", "sentiment": "positive"},
        {"review_text": "Decent watch for weekend.", "rating": "7/10", "author": "Reviewer", "sentiment": "neutral"},
        {"review_text": "Not worth the time.", "rating": "3/10", "author": "MovieCritic", "sentiment": "negative"}
    ]

def scrape_imdb_reviews_fast(imdb_id, max_reviews=25):
    """Optimized IMDb scraping with timeout and limits"""
    if not imdb_id:
        logging.warning("No IMDb ID provided for scraping reviews.")
        return {'reviews': [], 'poster_url': None}
    
    reviews_list = []
    poster_url = None
    url = f"[https://www.imdb.com/title/](https://www.imdb.com/title/){imdb_id}/reviews/"
    
    try:
        # Use direct requests.get() instead of session.get()
        response = requests.get(url, headers=movie_fetcher.headers, timeout=6)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find reviews with optimized selectors
        review_containers = (
            soup.find_all('article', class_=lambda x: x and 'user-review-item' in x) or
            soup.find_all('article', attrs={'data-testid': lambda x: x and 'review' in x}) or
            soup.find_all('article')[:20]
        )
        
        for container in review_containers[:20]:
            # Streamlined element finding
            review_text_el = (
                container.find('div', class_=lambda x: x and 'content' in str(x)) or
                container.find('div', attrs={'data-testid': 'review-content'})
            )
            
            rating_el = container.find('span', class_=lambda x: x and 'rating' in str(x))
            author_el = container.find('span', class_=lambda x: x and 'author' in str(x))
            
            review_text = review_text_el.get_text(strip=True) if review_text_el else None
            
            if review_text and len(review_text) > 20:
                reviews_list.append({
                    'review_text': review_text[:500],
                    'rating': rating_el.get_text(strip=True) if rating_el else 'N/A',
                    'author': author_el.get_text(strip=True) if author_el else 'Anonymous',
                    'date': 'N/A'
                })
        
        # Quick poster grab
        try:
            poster_img = soup.find('img', {'data-testid': 'hero-media__poster'})
            if poster_img and poster_img.get('src'):
                original_url = poster_img['src']
                if 'amazon.com' in original_url:
                    poster_url = original_url.split('._V1_')[0] + '._V1_.jpg'
                else:
                    poster_url = original_url
        except Exception as e: # Added specific exception logging
            logging.warning(f"Failed to scrape poster URL: {e}")
            pass
            
    except requests.exceptions.RequestException as req_e:
        logging.error(f"IMDb scraping request error for {imdb_id}: {req_e}")
    except Exception as e:
        logging.error(f"IMDb scraping parsing error for {imdb_id}: {e}")
    
    return {'reviews': reviews_list, 'poster_url': poster_url}

# Initialize optimized fetcher
movie_fetcher = OptimizedMovieFetcher() # This will now use direct requests

@app.route('/health')
def health():
    return "OK", 200

@app.route('/fetch_movie_data', methods=['POST'])
def fetch_movie_data():
    """Optimized main endpoint with parallel processing"""
    try:
        data = request.get_json()
        movie_query = data.get('query')
        
        if not movie_query:
            logging.warning("No movie query provided in fetch_movie_data request.")
            return jsonify({'error': 'No movie query provided'}), 400
        
        logging.info(f"Received request for movie: {movie_query}")
        
        # Step 1: Correct movie name (if Gemini available)
        corrected_query = find_movie_name_optimized(movie_query)
        logging.info(f"Corrected movie query: {corrected_query}")
        
        # Step 2: Get movie details from OMDB
        movie_details = movie_fetcher.get_movie_details_omdb(title=corrected_query)
        
        if not movie_details:
            logging.warning(f"Movie details not found for '{corrected_query}'.")
            return jsonify({'error': f"Movie not found: '{movie_query}'"}), 404
        
        logging.info(f"Found movie details for: {movie_details.get('title')}")
        
        # Step 3: Parallel processing for reviews and additional data
        reviews = []
        imdb_id = movie_details.get('imdb_id')
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks concurrently
            futures = []
            
            if imdb_id:
                logging.info(f"Submitting IMDb scraping for IMDb ID: {imdb_id}")
                futures.append(executor.submit(scrape_imdb_reviews_fast, imdb_id))
            
            logging.info(f"Submitting Gemini review generation for: {movie_details.get('title', corrected_query)}")
            futures.append(executor.submit(generate_fast_reviews, movie_details.get('title', corrected_query)))
            
            # Collect results
            scraped_data = None
            generated_reviews = None
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, dict) and 'reviews' in result:
                        scraped_data = result
                        logging.info("Collected scraped reviews and poster data.")
                    elif isinstance(result, list):
                        generated_reviews = result
                        logging.info("Collected generated reviews.")
                except Exception as e:
                    logging.error(f"Future execution error in fetch_movie_data: {e}")
        
        # Use scraped reviews if available, otherwise use generated ones
        if scraped_data and scraped_data.get('reviews'):
            reviews = scraped_data['reviews']
            if scraped_data.get('poster_url'):
                movie_details['poster_url'] = scraped_data['poster_url']
            logging.info(f"Using {len(reviews)} scraped reviews.")
        elif generated_reviews:
            reviews = generated_reviews
            logging.info(f"Using {len(reviews)} generated reviews.")
        else:
            reviews = get_static_reviews()
            logging.info(f"Falling back to static reviews.")
        
        return jsonify({
            'details': movie_details,
            'reviews': reviews[:10], # Limit to 10 reviews max for safety
            'source': 'Educational Project'
        })
        
    except json.JSONDecodeError:
        logging.error("Invalid JSON in request to /fetch_movie_data.")
        return jsonify({'error': 'Invalid JSON format in request'}), 400
    except Exception as e:
        logging.error(f"Unhandled error in fetch_movie_data: {e}", exc_info=True) # exc_info to print traceback
        return jsonify({'error': 'Internal server error processing movie data'}), 500


@app.route('/dummy-reviews', methods=['POST'])
def fetch_dummy_reviews():
    """
    Generates a list of 10 short dummy movie user reviews with mixed sentiment
    using the Gemini 1.5 Flash model and returns them as a JSON array.
    """
    try:
        # Initialize the generative model
        llm_model = genai.GenerativeModel('gemini-1.5-flash')
        chat = llm_model.start_chat()

        prompt = (
            "Generate a JSON list of 10 short dummy movie user reviews. "
            "Each review should be 1–2 sentences long. "
            "The reviews must be **mixed** in sentiment: include positive, neutral, and negative reviews. "
            "Ensure the mix is different for each request (e.g., 4 good, 3 bad, 3 neutral; or 5 bad, 2 good, 3 neutral, etc.). "
            "Format strictly as a JSON array of strings, e.g.: "
            '["Great movie!", "Bad plot.", "Okay to watch once.", "Waste of time", "Loved the acting", ...]. '
            "No explanations, just return the array."
        )

        max_retries = 3
        retries = 0
        reviews = None

        while retries < max_retries:
            try:
                response = chat.send_message(prompt)
                raw_text = response.text.strip()

                # Remove markdown backticks if present (e.g., ```json ... ```)
                if raw_text.startswith('```json'):
                    raw_text = raw_text[7:]
                if raw_text.endswith('```'):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                # Parse JSON array
                reviews = json.loads(raw_text)

                if not isinstance(reviews, list):
                    raise ValueError("Response is not a JSON list.")

                break # Success, exit the retry loop

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error (attempt {retries + 1}): {e}. Raw text: '{raw_text}'")
            except ValueError as ve:
                logging.error(f"Validation error (attempt {retries + 1}): {ve}")
            except Exception as e:
                logging.error(f"Unexpected error during Gemini API call (attempt {retries + 1}): {e}")

            retries += 1
            time.sleep(2 ** retries) # Exponential backoff

        if reviews:
            # Return the generated reviews as a JSON response
            return jsonify({'reviews': reviews, 'sentiment_classification': 'mixed', 'descriptive_passage': 'This is a dummy passage.', 'details': {}}), 200
        else:
            # If after retries, no valid reviews are obtained
            logging.error(f"Failed to generate dummy reviews after {max_retries} attempts.")
            return jsonify({'error': f'Failed to generate dummy reviews after {max_retries} attempts.'}), 500

    except Exception as e:
        logging.error(f"Unhandled error in fetch_dummy_reviews: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)