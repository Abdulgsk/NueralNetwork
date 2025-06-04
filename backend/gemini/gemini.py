import logging
import requests
import json
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from dotenv import load_dotenv
import os
from bs4 import BeautifulSoup  # Added missing import

logging.basicConfig(
    level=logging.WARNING,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)

load_dotenv()

app = Flask(__name__)
CORS(app)
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
        except Exception as e:
            logging.error(f"TMDB API error: {e}")
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
        except Exception as e:
            logging.error(f"Error getting IMDB ID: {e}")
        return None
    
    def get_movie_details_omdb(self, imdb_id=None, title=None):
        """Get movie details from OMDB API"""
        if not OMDB_API_KEY:
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
                logging.info(f"OMDB Response: {data}")
                
                # Helper function to convert string to int safely
                def safe_int(value):
                    if value and value != 'N/A':
                        try:
                            return int(value.replace(',', '').replace('$', ''))
                        except (ValueError, AttributeError):
                            pass
                    return None
                
                # Helper function to convert string to float safely
                def safe_float(value):
                    if value and value != 'N/A':
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            pass
                    return None
                
                # Helper function to split comma-separated strings into lists
                def split_to_list(value):
                    if value and value != 'N/A':
                        return [item.strip() for item in value.split(',') if item.strip()]
                    return []
                
                # Helper function to parse runtime to minutes
                def parse_runtime(runtime_str):
                    if runtime_str and runtime_str != 'N/A':
                        try:
                            # Extract numbers from string like "137 min"
                            import re
                            numbers = re.findall(r'\d+', runtime_str)
                            if numbers:
                                return int(numbers[0])
                        except (ValueError, AttributeError):
                            pass
                    return None
                def transform_ratings(ratings_list):
                    """Transform ratings from OMDB format to component format"""
                    if not ratings_list: return []
        
                    transformed = []
                    for rating in ratings_list:
                        transformed.append({
                        'source': rating.get('Source'),
                        'value': rating.get('Value')
                    })
                    return transformed
                # Helper function to parse box office
                def parse_box_office(box_office_str):
                    if box_office_str and box_office_str != 'N/A':
                        try:
                            # Remove currency symbols and commas, then convert to int
                            cleaned = box_office_str.replace('$', '').replace(',', '')
                            return int(cleaned)
                        except (ValueError, AttributeError):
                            pass
                    return None
                
                # Parse ratings into structured format
                ratings = {}
                if data.get('imdbRating') and data.get('imdbRating') != 'N/A':
                    ratings['imdb'] = safe_float(data.get('imdbRating'))
                
                # Parse Ratings array for Rotten Tomatoes and Metacritic
                if data.get('Ratings'):
                    for rating in data.get('Ratings', []):
                        source = rating.get('Source', '').lower()
                        value = rating.get('Value', '')
                        
                        if 'rotten tomatoes' in source:
                            ratings['rotten_tomatoes'] = value
                        elif 'metacritic' in source:
                            # Extract number from "84/100" format
                            try:
                                ratings['metacritic'] = int(value.split('/')[0])
                            except (ValueError, IndexError):
                                ratings['metacritic'] = value
                
                return {
                    'title': data.get('Title'),
                    'year': safe_int(data.get('Year')),
                    'rated': data.get('Rated') if data.get('Rated') != 'N/A' else None,
                    'released_date': data.get('Released') if data.get('Released') != 'N/A' else None,
                    'runtime_minutes': parse_runtime(data.get('Runtime')),
                    'genres': split_to_list(data.get('Genre')),
                    'director': data.get('Director') if data.get('Director') != 'N/A' else None,
                    'writer': data.get('Writer') if data.get('Writer') != 'N/A' else None,
                    'actors': split_to_list(data.get('Actors')),
                    'plot': data.get('Plot') if data.get('Plot') != 'N/A' else None,
                    'languages': split_to_list(data.get('Language')),
                    'countries': split_to_list(data.get('Country')),
                    'awards': data.get('Awards') if data.get('Awards') != 'N/A' else None,
                    'poster_url': data.get('Poster') if data.get('Poster') != 'N/A' else None,
                    'ratings': transform_ratings(data.get('Ratings')), 
                    'imdb_id': data.get('imdbID'),
                    'imdb_votes': data.get('imdbVotes') if data.get('imdbVotes') != 'N/A' else None,
                    'type': data.get('Type'),
                    'box_office': parse_box_office(data.get('BoxOffice')),
                    'production': data.get('Production') if data.get('Production') != 'N/A' else None,
                    'website': data.get('Website') if data.get('Website') != 'N/A' else None,
                    'dvd_release': data.get('DVD') if data.get('DVD') != 'N/A' else None,
                    'metascore': safe_int(data.get('Metascore')),
                    'response': data.get('Response'),
                    # Include any additional fields that might be present
                    'total_seasons': safe_int(data.get('totalSeasons')) if data.get('Type') == 'series' else None
                }
        except Exception as e:
            logging.error(f"OMDB API error: {e}")
        return None

def find_movie_name(movie_query):
    """
    Attempts to find the correct spelling/official title of a movie
    given a potentially misspelled or partial query, using Gemini AI.

    Args:
        movie_query (str): The user's input movie title (can be misspelled).

    Returns:
        str: The most likely correct movie title, or the original query
            if a definitive correction can't be determined or an error occurs.
    """
    # Placeholder for GEMINI_API_KEY. Replace with your actual key or load from environment.
    # For this function to work, genai.configure(api_key=YOUR_API_KEY) must be called elsewhere.
    if not GEMINI_API_KEY:
        logging.warning("GEMINI_API_KEY not set. Using fallback logic.")
        return movie_query  

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        Given the following movie title query, identify its most likely correct and official spelling.
        If the query seems correct, return it as is. If it's a common misspelling or partial name,
        provide the full and correct title. Prioritize exact matches if possible.

        Query: "{movie_query}"

        Return only the corrected movie title as a plain string, without any additional text or JSON formatting.
        For example:
        Input: "avilars"
        Output: "Avatar"

        Input: "Spider-Man Far From Home hollywod marvel"
        Output: "Spider-Man: Far From Home"
      
        Input: "The Shawshank Redemtion morgan freeman"    
        Output: "The Shawshank Redemption"

        Input: "The Dark Knight chrishtofher nolan"
        Output: "The Dark Knight"
        
        Input: "RRR telugu movie"
        Output: "RRR"

        Make sure the name is the official title as recognized by major movie databases like IMDb or TMDB.
        If you cannot determine a correction, return the original query as is.

        """

        response = model.generate_content(prompt)
        corrected_movie_name = response.text.strip()

        # Basic validation: ensure the response isn't unexpectedly empty or very short
        if corrected_movie_name and len(corrected_movie_name) > 1:
            return corrected_movie_name
        else:
            logging.warning(f"Gemini returned an ambiguous or empty response for '{movie_query}'. Returning original query.")
            return movie_query

    except Exception as e:
        logging.error(f"Error finding movie name with Gemini for query '{movie_query}': {str(e)}")
        return movie_query # Fallback: return the original query on error # Fallback: return the original query on error

def generate_sample_reviews(movie_title, count=10):
    """Generate sample reviews using Gemini AI"""
    if not GEMINI_API_KEY:
        return generate_static_reviews()
    
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
        logging.error(f"Error generating reviews with Gemini: {str(e)}")
    return generate_static_reviews()

def generate_static_reviews():
    """Fallback static reviews for testing"""
    return [
        {"review_text": "Great movie! Really enjoyed the storyline and acting.", "rating": "8/10", "author": "CinemaLover", "sentiment": "positive"},
        {"review_text": "Not bad, but could have been better. Average entertainment.", "rating": "6/10", "author": "MovieCritic99", "sentiment": "neutral"},
        {"review_text": "Disappointing. Expected much more from this film.", "rating": "4/10", "author": "FilmBuff", "sentiment": "negative"},
        {"review_text": "Absolutely loved it! One of the best movies I've seen this year.", "rating": "9/10", "author": "ReviewGuru", "sentiment": "positive"},
        {"review_text": "Decent watch. Good for a lazy Sunday afternoon.", "rating": "7/10", "author": "WeekendViewer", "sentiment": "neutral"}
    ]

def scrape_imdb_reviews(imdb_id):
    """
    Scrapes user reviews from IMDb for a given IMDb ID.
    Updated selectors based on current IMDb HTML structure (2025).
    """
    if not imdb_id:
        logging.warning("No IMDb ID provided for review scraping")
        return []
        
    reviews_list = []
    # Base URL for reviews
    url = f"https://www.imdb.com/title/{imdb_id}/reviews/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36"
    }

    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=15)  # Increase timeout to 15
                break
            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
        time.sleep(2)
        logging.info(f"Attempting to scrape reviews from: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Updated selectors based on current IMDb structure from the screenshot
        # Looking for articles with class containing user-review-item
        review_containers = soup.find_all('article', class_=lambda x: x and 'user-review-item' in x)
        
        if not review_containers:
            logging.warning(f"No review containers found with 'user-review-item' class. Trying alternative selectors...")
            # Try alternative selectors based on the HTML structure seen
            review_containers = soup.find_all('article', attrs={'data-testid': lambda x: x and 'review' in x})
            
        if not review_containers:
            # Try more generic article tags that might contain reviews
            review_containers = soup.find_all('article')
            
        if not review_containers:
            logging.warning(f"No review containers found for {imdb_id}. HTML structure might have changed.")
            return []

        for container in review_containers:
            # Based on the HTML structure, look for review content
            # The review text is likely in a div with specific classes
            review_text_el = (
                container.find('div', class_=lambda x: x and 'ipc-list-card__content' in x) or
                container.find('div', class_=lambda x: x and 'content' in x) or
                container.find('div', attrs={'data-testid': 'review-content'}) or
                container.find('div', class_='text')
            )
            
            # Look for rating - might be in span with rating classes
            rating_el = (
                container.find('span', class_=lambda x: x and 'rating' in x) or
                container.find('div', class_=lambda x: x and 'rating' in x) or
                container.find('span', class_='point-scale')
            )
            
            # Look for author name
            author_el = (
                container.find('span', class_=lambda x: x and 'author' in x) or
                container.find('a', class_=lambda x: x and 'author' in x) or
                container.find('span', attrs={'data-testid': 'author'}) or
                container.find('span', class_='display-name-link')
            )
            
            # Look for date
            date_el = (
                container.find('span', class_=lambda x: x and 'date' in x) or
                container.find('time') or
                container.find('span', class_='review-date')
            )

            # Extract text content
            review_text = review_text_el.get_text(strip=True) if review_text_el else None
            rating = rating_el.get_text(strip=True) if rating_el else 'N/A'
            author = author_el.get_text(strip=True) if author_el else 'Anonymous'
            date = date_el.get_text(strip=True) if date_el else 'N/A'

            # Only add if we have actual review text and it's substantial
            if review_text and len(review_text) > 10:
                reviews_list.append({
                    'review_text': review_text,
                    'rating': rating,
                    'author': author,
                    'date': date
                })
        
        logging.info(f"Scraped {len(reviews_list)} reviews for {imdb_id}.")
        
        # If we didn't get any reviews, log the page structure for debugging
        if not reviews_list:
            logging.warning("No reviews found. Checking page structure...")
            articles = soup.find_all('article')
            logging.info(f"Found {len(articles)} article tags on the page")
            if articles:
                # Log the class names of the first few articles for debugging
                for i, article in enumerate(articles[:3]):
                    logging.info(f"Article {i} classes: {article.get('class', [])}")
                    
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP/Network error during scraping for {imdb_id}: {e}")
    except Exception as e:
        logging.error(f"Error parsing HTML or extracting data for {imdb_id}: {e}")
    
    # Add this code before the return statement in scrape_imdb_reviews function
    # Replace the existing poster scraping code with this:
    poster_url = None
    try:
        # Look for the poster image
        poster_img = soup.find('img', {'data-testid': 'hero-media__poster'}) or \
                    soup.find('img', class_=lambda x: x and 'poster' in str(x).lower()) or \
                    soup.find('div', class_='ipc-media').find('img') if soup.find('div', class_='ipc-media') else None
        
        if poster_img and poster_img.get('src'):
            original_url = poster_img['src']
            
            # Transform the URL to get high-resolution version
            # IMDb image URLs often have size parameters that can be modified
            if 'amazon.com' in original_url:
                # Remove size constraints and get original size
                # Replace common size patterns with high-res versions
                poster_url = original_url.split('._V1_')[0] + '._V1_.jpg'
                # Alternative: You can also try specific dimensions
                # poster_url = original_url.split('._V1_')[0] + '._V1_SX1000_.jpg'
            else:
                poster_url = original_url
                
            logging.info(f"Scraped poster URL: {poster_url}")
    except Exception as e:
        logging.error(f"Error scraping poster: {e}")

    time.sleep(1)
    return {'reviews': reviews_list, 'poster_url': poster_url}
    # Add a small delay between requests

# Initialize movie fetcher
movie_fetcher = MovieDataFetcher()

@app.route('/health')
def health():
    return "OK", 200

@app.route('/fetch_movie_data', methods=['POST'])
def fetch_movie_data():
    try:
        data = request.get_json()
        movie_query = data.get('query')
        
        try:
            movie_query = find_movie_name(movie_query)
        except Exception as e:
            logging.error(f"Gemini API error: {str(e)}")
            return jsonify({'error': 'Movie search failed. Please enter the exact movie title and try again.'}), 400
        
        if not movie_query:
            return jsonify({'error': 'No movie query provided'}), 400
        
        logging.info(f"Searching for movie: {movie_query}")
        
        # Try OMDB first
        movie_details = movie_fetcher.get_movie_details_omdb(title=movie_query)
        
        # If OMDB doesn't work, try OMDB
        if not movie_details:
            movie_details = movie_fetcher.search_movie_tmdb(movie_query)
        
        if not movie_details:
            return jsonify({'error': f"Enter the correct details or be more specific '{movie_query}'"}), 404
        
        # Scrape reviews using IMDb ID if available
        reviews = []
        imdb_id = movie_details.get('imdb_id')

        # With this:
        if imdb_id:
            scraped_data = scrape_imdb_reviews(imdb_id)
            reviews = scraped_data.get('reviews', [])
            
            # Override OMDB poster with scraped poster if available
            if scraped_data.get('poster_url'):
                movie_details['poster_url'] = scraped_data['poster_url']
        
        # If no reviews scraped, generate sample reviews as fallback
        if not reviews:
            logging.info("No reviews scraped, generating sample reviews")
            reviews = generate_sample_reviews(movie_details.get('title', movie_query))
        
        return jsonify({
            'details': movie_details,
            'reviews': reviews,
            'source': 'Educational/Learning Project - Sample Data'
        })
        
    except Exception as e:
        logging.error(f"Error in fetch_movie_data: {e}")
        return jsonify({'error': str(e)}), 500

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

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)