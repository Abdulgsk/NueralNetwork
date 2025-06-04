import logging
import requests
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure APIs
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TMDB_API_KEY = os.getenv("TMDB_API_KEY")  # Get free API key from themoviedb.org
OMDB_API_KEY = os.getenv("OMDB_API_KEY")  # Get free API key from omdbapi.com

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

class MovieDataFetcher:
    def __init__(self):
        self.tmdb_base_url = "https://api.themoviedb.org/3"
        self.omdb_base_url = "http://www.omdbapi.com"
    
    def search_movie_tmdb(self, query):
        """Search for movie using TMDB API"""
        if not TMDB_API_KEY:
            logging.warning("TMDB API key not found.")
            return None
            
        try:
            url = f"{self.tmdb_base_url}/search/movie"
            params = {
                'api_key': TMDB_API_KEY,
                'query': query,
                'language': 'en-US'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['results']:
                movie = data['results'][0]  # Get first result
                return {
                    'title': movie.get('title'),
                    'year': movie.get('release_date', '').split('-')[0] if movie.get('release_date') else 'N/A',
                    'overview': movie.get('overview'),
                    'rating': movie.get('vote_average'),
                    'poster_path': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie.get('poster_path') else None,
                    'tmdb_id': movie.get('id'),
                    'imdb_id': self.get_imdb_id_from_tmdb(movie.get('id'))
                }
        except requests.exceptions.RequestException as e:
            logging.error(f"TMDB API request error for query '{query}': {e}")
        except Exception as e:
            logging.error(f"Error processing TMDB response for query '{query}': {e}")
        return None
    
    def get_imdb_id_from_tmdb(self, tmdb_id):
        """Get IMDB ID from TMDB"""
        if not TMDB_API_KEY or not tmdb_id:
            return None
            
        try:
            url = f"{self.tmdb_base_url}/movie/{tmdb_id}/external_ids"
            params = {'api_key': TMDB_API_KEY}
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return data.get('imdb_id')
        except requests.exceptions.RequestException as e:
            logging.error(f"Error getting IMDB ID from TMDB for ID '{tmdb_id}': {e}")
        except Exception as e:
            logging.error(f"Error processing TMDB external IDs for ID '{tmdb_id}': {e}")
        return None
    
    def get_movie_details_omdb(self, imdb_id=None, title=None):
        """Get movie details from OMDB API"""
        if not OMDB_API_KEY:
            logging.warning("OMDB API key not found.")
            return None
            
        try:
            params = {'apikey': OMDB_API_KEY}
            
            if imdb_id:
                params['i'] = imdb_id
            elif title:
                params['t'] = title
            else:
                return None
                
            response = requests.get(self.omdb_base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('Response') == 'True':
                logging.info(f"OMDB Response for '{imdb_id or title}': {data.get('Title')}")
                return {
                    'title': data.get('Title'),
                    'year': data.get('Year'),
                    'plot': data.get('Plot'),
                    'director': data.get('Director'),
                    'actors': data.get('Actors'),
                    'genre': data.get('Genre'),
                    'imdb_rating': data.get('imdbRating'),
                    'imdb_votes': data.get('imdbVotes'),
                    'poster': data.get('Poster') if data.get('Poster') != 'N/A' else None,
                    'imdb_id': data.get('imdbID'),
                    'runtime': data.get('Runtime'),
                    'country': data.get('Country'),
                    'language': data.get('Language'),
                    'awards': data.get('Awards')
                }
        except requests.exceptions.RequestException as e:
            logging.error(f"OMDB API request error for '{imdb_id or title}': {e}")
        except Exception as e:
            logging.error(f"Error processing OMDB response for '{imdb_id or title}': {e}")
        return None

def correct_movie_query_with_gemini(query):
    """
    Uses Gemini AI to correct or refine a movie query, ensuring it's a valid movie title.
    Returns the corrected title or the original query if Gemini API is not configured or fails.
    """
    if not GEMINI_API_KEY:
        logging.warning("Gemini API key not found for query correction. Skipping.")
        return query
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        The user is trying to find a movie. Their input is "{query}".
        Identify the most likely correct and official movie title from this input.
        If it's already a well-known movie title, just return it.
        If it's misspelled or unclear, try to infer the correct title.
        If it seems like a general phrase or doesn't resemble a movie title, just return the original input.
        Do NOT add any extra text, explanations, or formatting (like quotes or bullet points).
        Just respond with the clean, corrected movie title.

        Examples:
        Input: "Dark Knight Rises"
        Output: The Dark Knight Rises

        Input: "spiderman far from home"
        Output: Spider-Man: Far From Home

        Input: "movy"
        Output: movy

        Input: "action movie"
        Output: action movie
        """
        
        response = model.generate_content(prompt)
        corrected_title = response.text.strip()
        logging.info(f"Gemini corrected '{query}' to '{corrected_title}'")
        return corrected_title
        
    except Exception as e:
        logging.error(f"Error correcting movie query with Gemini for '{query}': {e}")
        return query # Fallback to original query on error

def scrape_imdb_reviews(imdb_id):
    """
    Scrapes user reviews from IMDb for a given IMDb ID.
    Updated selectors based on current IMDb HTML structure (2025).
    """
    if not imdb_id:
        logging.warning("No IMDb ID provided for review scraping.")
        return []
        
    reviews_list = []
    url = f"https://www.imdb.com/title/{imdb_id}/reviews/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }

    try:
        logging.info(f"Attempting to scrape reviews from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Prioritize data-testid attribute for robustness
        review_containers = soup.find_all('div', attrs={'data-testid': lambda x: x and 'review-card' in x})
        
        if not review_containers:
            logging.warning(f"No review containers found with 'data-testid' for {imdb_id}. Trying alternative selectors...")
            # Fallback to article tags with specific classes
            review_containers = soup.find_all('article', class_=lambda x: x and 'user-review-item' in x)
        
        if not review_containers:
            logging.warning(f"No review containers found for {imdb_id} using any known selectors. HTML structure might have changed significantly.")
            return []

        for container in review_containers:
            review_text_el = (
                container.find('div', class_='text show-more__control') or # Common text class
                container.find('div', attrs={'data-testid': 'review-content'}) or
                container.find('div', class_='content').find('div', class_='text') # More nested cases
            )
            
            rating_el = (
                container.find('span', class_='rating-other-user-rating') or # Specific rating span
                container.find('span', class_=lambda x: x and 'rating' in x) # More generic rating class
            )
            
            author_el = (
                container.find('span', class_='display-name-link') or # Author link
                container.find('a', class_=lambda x: x and 'author' in x)
            )
            
            date_el = (
                container.find('span', class_='review-date') or # Specific date span
                container.find('time')
            )

            review_text = review_text_el.get_text(strip=True) if review_text_el else None
            rating = rating_el.get_text(strip=True) if rating_el else 'N/A'
            author = author_el.get_text(strip=True) if author_el else 'Anonymous'
            date = date_el.get_text(strip=True) if date_el else 'N/A'

            if review_text and len(review_text) > 10: # Ensure review is substantial
                reviews_list.append({
                    'review_text': review_text,
                    'rating': rating,
                    'author': author,
                    'date': date
                })
        
        logging.info(f"Scraped {len(reviews_list)} reviews for {imdb_id}.")
        
        # If no reviews found, log some page structure for debugging
        if not reviews_list and review_containers:
            logging.info("No reviews found despite finding containers. Logging a sample container HTML for debugging:")
            logging.info(review_containers[0].prettify())

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP/Network error during IMDb scraping for {imdb_id}: {e}")
    except Exception as e:
        logging.error(f"Error parsing HTML or extracting data from IMDb for {imdb_id}: {e}")
    
    time.sleep(1) # Be polite to the server
    return reviews_list

# Initialize movie fetcher
movie_fetcher = MovieDataFetcher()

@app.route('/health')
def health():
    return "OK", 200

@app.route('/fetch_movie_data', methods=['POST'])
def fetch_movie_data():
    try:
        data = request.get_json()
        raw_movie_query = data.get('query')
        
        if not raw_movie_query:
            return jsonify({'error': 'No movie query provided'}), 400
        
        logging.info(f"Received raw movie query: '{raw_movie_query}'")

        # Step 1: Use Gemini to get a more accurate movie title
        corrected_movie_query = correct_movie_query_with_gemini(raw_movie_query)
        logging.info(f"Using corrected movie query: '{corrected_movie_query}' for API calls.")
        
        tmdb_details = None
        omdb_details = None
        
        # Step 2: Try TMDB first with the corrected query
        tmdb_details = movie_fetcher.search_movie_tmdb(corrected_movie_query)
        
        # Step 3: Use the IMDB ID from TMDB to get OMDB details
        imdb_id_from_tmdb = tmdb_details.get('imdb_id') if tmdb_details else None
        if imdb_id_from_tmdb:
            omdb_details = movie_fetcher.get_movie_details_omdb(imdb_id=imdb_id_from_tmdb)
        else:
            # If no IMDB ID from TMDB, try OMDB directly with the corrected title
            omdb_details = movie_fetcher.get_movie_details_omdb(title=corrected_movie_query)
        
        # Combine details from both sources
        combined_details = {}
        if tmdb_details:
            combined_details.update(tmdb_details)
        if omdb_details:
            combined_details.update(omdb_details) # OMDB details can overwrite/complement TMDB details

        if not combined_details:
            return jsonify({'error': f"Could not find movie details for '{raw_movie_query}'. Please try a different title."}), 404
        
        # Step 4: Scrape reviews using IMDb ID
        final_imdb_id = combined_details.get('imdb_id')
        reviews = []
        if final_imdb_id:
            reviews = scrape_imdb_reviews(final_imdb_id)
        else:
            logging.warning("No IMDb ID available to scrape reviews.")
        
        # If no reviews scraped, return an empty list for reviews
        if not reviews:
            logging.info(f"No reviews scraped for '{combined_details.get('title', corrected_movie_query)}'.")
        
        return jsonify({
            'details': combined_details,
            'reviews': reviews,
            'source': 'Combined TMDB/OMDB and Scraped IMDb Reviews'
        })
            
    except Exception as e:
        logging.error(f"Error in fetch_movie_data: {e}")
        return jsonify({'error': str(e)}), 500

def generate_sample_reviews(movie_title, count=10):
    """Generate sample reviews using Gemini AI"""
    if not GEMINI_API_KEY:
        return logging.error(f"GEMINI API KEY no provided: {e}")
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        Generate {count} realistic movie reviews for "{movie_title}". 
        Make them diverse in sentiment (positive, negative, neutral) and writing style.
        Each review should be 1-3 sentences long and feel authentic.
        Include a mix of detailed and brief reviews.
        
        Return as a JSON array of objects with this format:
        [{{"review_text": "...", "rating": "7/10", "author": "MovieFan123", "sentiment": "positive"}}]
        
        Make the reviews feel like real user opinions, not promotional content.
        """
        
        response = model.generate_content(prompt)
        reviews_text = response.text.strip()
        
        # Clean JSON response
        if reviews_text.startswith('```json'):
            reviews_text = reviews_text[7:]
        if reviews_text.endswith('```'):
            reviews_text = reviews_text[:-3]
        reviews_text = reviews_text.strip()
        
        reviews = json.loads(reviews_text)
        return reviews if isinstance(reviews, list) else []
        
    except Exception as e:
        logging.error(f"Error generating reviews with Gemini: {e}")

@app.route('/fetch_dummy_reviews', methods=['POST'])
def fetch_dummy_reviews():
    """Generate dummy reviews for testing"""
    try:
        data = request.get_json()
        movie_title = data.get('title', 'Sample Movie')
        count = data.get('count', 10)
        
        reviews = generate_sample_reviews(movie_title, count)
        return jsonify({'reviews': reviews})
        
    except Exception as e:
        logging.error(f"Error generating dummy reviews: {e}")
        return jsonify({'error': str(e)}), 500

# Remove the /fetch_dummy_reviews endpoint as it's no longer needed based on requirements
# @app.route('/fetch_dummy_reviews', methods=['POST'])
# def fetch_dummy_reviews():
#     """Generate dummy reviews for testing - REMOVED PER NEW REQUIREMENTS"""
#     pass

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)