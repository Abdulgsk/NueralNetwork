# gemini.py
import logging
logging.basicConfig(
    level=logging.INFO,  # You can change to DEBUG for more detailed output
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import json
import time # Import time for delays
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure Gemini API key for this specific app
# Make sure to replace 'YOUR_GEMINI_API_KEY_HERE' with your actual API key
api_key = os.getenv("GEMINI_API_KEY")
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY is not set in environment variables.")
    raise RuntimeError("GEMINI_API_KEY is required but not set.")
genai.configure(api_key=api_key)


@app.route('/health')
def health():
    return "OK", 200

@app.route('/fetch_movie_data', methods=['POST'])
def fetch_movie_data():
    try:
        data = request.get_json()
        movie_query = data.get('query')

        if not movie_query:
            return jsonify({'error': 'No movie query provided'}), 400

        llm_model = genai.GenerativeModel('gemini-1.5-flash')
        chat = llm_model.start_chat()

        prompt = (
            f"Please provide the movie title, year, genre, and a list of exactly 20 common user reviews (short and concise, 1-2 sentences each, from online resources) for the movie: '{movie_query}'. "
            "If you cannot find specific reviews, return an empty list for 'reviews'. "
            "Format the output strictly as a JSON object with two top-level keys: 'details' and 'reviews'. "
            "The 'details' object should have 'title', 'year', 'genre' , 'director' and 'cast'keys. "
            "Example JSON: {'details': {'title': 'Inception', 'year': 2010, 'genre': 'Sci-Fi', 'director': 'David-Fincher', 'cast':'Tom crusie, hugh jaksman'}, 'reviews': ['Great movie!', 'Mind-blowing concept.']}"
        )

        max_retries = 3
        retries = 0
        movie_data = None

        while retries < max_retries:
            try:
                logging.info(f"Sending prompt to Gemini (Attempt {retries + 1}/{max_retries}): {prompt}")
                response = chat.send_message(prompt)
                gemini_raw_text = response.text
                logging.info(f"Received from Gemini (raw, Attempt {retries + 1}): {gemini_raw_text[:500]}...")

                # Strip markdown code block markers if present
                gemini_raw_text = gemini_raw_text.strip()
                if gemini_raw_text.startswith('```json'):
                    gemini_raw_text = gemini_raw_text[7:]  # Remove ```json
                if gemini_raw_text.endswith('```'):
                    gemini_raw_text = gemini_raw_text[:-3]  # Remove ```
                gemini_raw_text = gemini_raw_text.strip()

                # Parse the cleaned response
                movie_data = json.loads(gemini_raw_text)

                # Validate the structure
                if "details" not in movie_data or "reviews" not in movie_data:
                    raise ValueError("Gemini response missing 'details' or 'reviews' key.")
                if not isinstance(movie_data.get("reviews"), list):
                    raise ValueError("Gemini response 'reviews' is not a list.")

                break  # If successful, exit the loop

            except json.JSONDecodeError as e:
                logging.error(f"Error parsing Gemini response as JSON (Attempt {retries + 1}): {e}")
                logging.error(f"Raw Gemini response that caused error: {gemini_raw_text[:500]}...")
                retries += 1
                time.sleep(2)
            except ValueError as ve:
                logging.error(f"Invalid structure from Gemini (Attempt {retries + 1}): {ve} - Raw Response: {gemini_raw_text[:500]}...")
                retries += 1
                time.sleep(2)
            except Exception as e:
                logging.error(f"Unexpected error from Gemini (Attempt {retries + 1}): {e}")
                retries += 1
                time.sleep(2)

        if movie_data:
            return jsonify(movie_data)
        else:
            return jsonify({
                'error': f'Failed to fetch and parse movie data after {max_retries} attempts. Last raw response: {gemini_raw_text[:200]}...'
            }), 500

    except Exception as e:
        logging.error(f"Error in gemini_movie_api: {e}")
        return jsonify({'error': f'Failed to fetch movie data: {str(e)}'}), 500

@app.route('/fetch_dummy_reviews', methods=['POST'])
def fetch_dummy_reviews():
    try:
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

                # Remove markdown backticks if present
                if raw_text.startswith('```json'):
                    raw_text = raw_text[7:]
                if raw_text.endswith('```'):
                    raw_text = raw_text[:-3]
                raw_text = raw_text.strip()

                # Parse JSON array
                reviews = json.loads(raw_text)

                if not isinstance(reviews, list):
                    raise ValueError("Response is not a JSON list.")

                break  # success

            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error (attempt {retries + 1}): {e}")
            except ValueError as ve:
                logging.error(f"Validation error (attempt {retries + 1}): {ve}")
            except Exception as e:
                logging.error(f"Unexpected error (attempt {retries + 1}): {e}")

            retries += 1
            time.sleep(2 ** retries)
        if reviews:
            return jsonify({'reviews': reviews})
        else:
            return jsonify({'error': f'Failed after {max_retries} attempts.'}), 500

    except Exception as e:
        logging.error(f"Error in fetch_dummy_reviews_from_flash: {e}")
        return jsonify({'error': str(e)}), 500

    
if __name__ == '__main__':
    port = int(os.getenv("GEMINI_PORT", 5001))
    app.run(host="0.0.0.0", port=port)
