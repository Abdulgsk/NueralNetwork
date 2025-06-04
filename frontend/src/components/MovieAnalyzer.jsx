import React, { useState } from "react";
import MovieDetailsCard from "./MovieDetailsCard";
import MovieReviewsCard from "./MovieReviewsCard";

function MovieAnalyzer() {
  const [reviewText, setReviewText] = useState("");
  const [backendSentiment, setBackendSentiment] = useState(null); // This will store what the backend actually sends
  const [frontendClassification, setFrontendClassification] = useState(null); // This will store your desired "old" classification
  const [descriptivePassage, setDescriptivePassage] = useState(null);
  const [movieDetails, setMovieDetails] = useState(null);
  const [movieReviews, setMovieReviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [loadingExample, setLoadingExample] = useState(false);
  const [error, setError] = useState(null);
  const [showInfo, setShowInfo] = useState(false);

  // --- NEW FUNCTION: Map backend sentiment to old frontend classification ---
  const mapBackendSentimentToOldClassification = (backendClassification) => {
    if (!backendClassification) return "Analysis Pending"; // Or a suitable default

    const lowerCaseClassification = backendClassification.toLowerCase();

    if (lowerCaseClassification.includes("overwhelmingly positive")) {
      return "Cinematic Masterpiece"; // Covers "Masterpiece" and "Overwhelmingly Positive"
    }
    if (
      lowerCaseClassification.includes("highly positive") ||
      lowerCaseClassification.includes("positive") // Covers "Positive", "Moderately Positive"
    ) {
      return "Worth Watching";
    }
    if (
      lowerCaseClassification.includes("moderately positive") ||
      lowerCaseClassification.includes("neutral") ||
      lowerCaseClassification.includes("mixed")
    ) {
      return "Mixed Feelings";
    }
    if (
      lowerCaseClassification.includes("no strong sentiment") ||
      lowerCaseClassification.includes("moderately negative") ||
      lowerCaseClassification.includes("negative")
    ) {
      return "Disappointing";
    }
    if (
      lowerCaseClassification.includes("highly negative") ||
      lowerCaseClassification.includes("overwhelmingly negative")
    ) {
      return "Waste of Time";
    }
    return "Unknown Sentiment"; // Fallback
  };
  // --- END NEW FUNCTION ---

  const handleSubmit = async () => {
    setBackendSentiment(null);
    setFrontendClassification(null); // Reset
    setDescriptivePassage(null);
    setMovieDetails(null);
    setMovieReviews([]);
    setError(null);

    if (!reviewText.trim()) {
      setError("Please enter a movie review or a movie name.");
      return;
    }

    setLoading(true);

    try {
      const res = await fetch(
        "https://nueralnetwork-production.up.railway.app/predict", // Ensure this matches your Flask backend port
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: reviewText }),
        }
      );

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(
          errorData.error || "Failed to get sentiment from backend."
        );
      }

      const data = await res.json();
      setBackendSentiment(data.sentiment_classification); // Store backend's original classification
      // Set the frontend display classification based on mapping
      setFrontendClassification(
        mapBackendSentimentToOldClassification(data.sentiment_classification)
      );
      setDescriptivePassage(data.descriptive_passage);

      if (data.details && Object.keys(data.details).length > 0) {
        setMovieDetails(data.details);
      } else {
        setMovieDetails(null);
      }

      // Ensure movieReviews is an array of objects with 'review_text'
      if (
        data.reviews &&
        Array.isArray(data.reviews) &&
        data.reviews.length > 0
      ) {
        // If reviews are already objects with review_text, use them directly
        // Otherwise, assume they are strings and convert to objects
        const formattedReviews = data.reviews.map((review) =>
          typeof review === "string"
            ? { review_text: review, author: "N/A", rating: "N/A", date: "N/A" }
            : review
        );
        setMovieReviews(formattedReviews);
      } else {
        setMovieReviews([]);
      }
    } catch (err) {
      console.error("Error during sentiment analysis:", err);
      setError(`Analysis failed: Enter the exact details or be more specific`);
    } finally {
      setLoading(false);
    }
  };

  // --- Sentiment Color and Gradient Mapping using frontendClassification ---
  const getSentimentColor = (classification) => {
    if (!classification) return "text-gray-800";
    if (classification === "Cinematic Masterpiece") {
      return "text-green-400"; // A vibrant green
    }
    if (classification === "Worth Watching") {
      return "text-lime-300"; // A lighter green
    }
    if (classification === "Mixed Feelings") {
      return "text-amber-400"; // An orange
    }
    if (classification === "Disappointing") {
      return "text-red-400"; // A lighter red
    }
    if (classification === "Waste of Time") {
      return "text-red-700"; // A darker red
    }
    return "text-gray-800"; // Default
  };

  const getSentimentGradient = (classification) => {
    if (!classification) return "from-gray-500 to-gray-600";
    if (classification === "Cinematic Masterpiece") {
      return "from-green-500 to-green-600"; // Green to slightly darker green
    }
    if (classification === "Worth Watching") {
      return "from-lime-400 to-lime-500"; // Light green to slightly darker light green
    }
    if (classification === "Mixed Feelings") {
      return "from-amber-400 to-orange-500"; // Orange to a slightly deeper orange
    }
    if (classification === "Disappointing") {
      return "from-red-400 to-red-500"; // Light red to a slightly deeper red
    }
    if (classification === "Waste of Time") {
      return "from-red-700 to-red-800"; // Dark red to even darker red
    }
    return "from-gray-500 to-gray-600"; // Default
  };
  // --- End Sentiment Color and Gradient Mapping ---

  const handleTryItOut = async () => {
    setLoadingExample(true);
    setError(null);
    // Reset all previous results
    setBackendSentiment(null);
    setFrontendClassification(null);
    setDescriptivePassage(null);
    setMovieDetails(null);
    setMovieReviews([]);

    try {
  // Call the predict endpoint with a sample movie name
  const res = await fetch(
    "https://astonishing-cat-production.up.railway.app/fetch_dummy_reviews", // Ensure this matches your Flask backend port
    {
      method: 'POST', // Assuming your /dummy-reviews endpoint expects a POST request
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ movie_name: 'The Matrix' }), // Example movie query
    }
  );

  if (!res.ok) {
    const errorData = await res.json();
    throw new Error(
      errorData.error || "Failed to get example sentiment from backend."
    );
  }

 
  const data = await res.json();

    // *** ADD THESE LINES TO PROCESS SENTIMENT AND DESCRIPTIVE PASSAGE FROM DUMMY DATA ***
    setBackendSentiment(data.sentiment_classification);
    setFrontendClassification(mapBackendSentimentToOldClassification(data.sentiment_classification));
    setDescriptivePassage(data.descriptive_passage);
    // You might also want to set movieDetails if the dummy-reviews endpoint provides them
    if (data.details && Object.keys(data.details).length > 0) {
        setMovieDetails(data.details);
    } else {
        setMovieDetails(null);
    }

  // Populate the input field with the first review if available
  if (data.reviews && data.reviews.length > 0) {
    // Assuming reviews are objects with 'review_text' from backend
    const formattedReviews = data.reviews.map((review) =>
        typeof review === "string"
            ? { review_text: review, author: "N/A", rating: "N/A", date: "N/A" }
            : review
    );
    setMovieReviews(formattedReviews);
  } else {
    setReviewText("No online reviews found for this example movie."); // Fallback text
  }
} catch (err) {
  console.error("Error during example sentiment analysis:", err);
  setError(`Example analysis failed: ${err.message}`);
} finally {
  setLoadingExample(false);
}
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <style>
        {`
          @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
          }
          @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
          }
          @keyframes fadeInScale {
            from { opacity: 0; transform: scale(0.95); }
            to { opacity: 1; transform: scale(1); }
          }
          @keyframes pulse-custom {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.8; }
          }
          @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          .animate-float { animation: float 6s ease-in-out infinite; }
          .animate-fadeInUp { animation: fadeInUp 0.6s ease-out; }
          .animate-fadeInScale { animation: fadeInScale 0.5s ease-out; }
          .animate-pulse-custom { animation: pulse-custom 2s ease-in-out infinite; }
          .animate-slideIn { animation: slideIn 0.3s ease-out; }
          .glass-effect {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
          }

          /* Fixed info-tooltip styles for all screen sizes */
          /* NOTE: These styles are now for the MODAL, not just a tooltip */
          .info-modal {
            position: fixed;
            z-index: 50; /* Higher than the backdrop (z-40) */
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: calc(100vw - 2rem);
            max-width: 28rem;
            max-height: calc(100vh - 4rem);
            overflow-y: auto;
            box-sizing: border-box;
            padding: 1rem;
            background: rgba(45, 50, 60, 0.95); /* A slightly darker, more opaque background for the modal content */
            border: 1px solid rgba(100, 116, 139, 0.5); /* Border matching your original tooltip */
            border-radius: 0.75rem; /* rounded-xl */
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 10px 10px -5px rgba(0, 0, 0, 0.4); /* shadow-2xl */
          }

          /* Remove the arrow for better mobile experience */
          .info-modal::after {
            display: none;
          }

          /* Ensure proper scrolling on small screens */
          @media (max-height: 600px) {
            .info-modal {
              max-height: calc(100vh - 2rem);
              top: 1rem;
              transform: translateX(-50%);
            }
          }

          /* Extra small screens */
          @media (max-width: 480px) {
            .info-modal {
              width: calc(100vw - 1rem);
              padding: 0.75rem;
            }
          }
        `}
      </style>

      {/* Conditional rendering for the overlay AND the modal itself */}
      {showInfo && (
        <>
          {/* Full-screen overlay with blur and dimming */}
          <div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40" // This creates the blurred and dimmed background
            onClick={() => setShowInfo(false)}
          />

          {/* Info Modal/Overlay (appears on top of the blurred backdrop) */}
          <div className="info-modal animate-slideIn">
            {" "}
            {/* Using the .info-modal class */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-cyan-400 font-bold text-sm sm:text-base flex items-center space-x-2">
                  <span>💡</span>
                  <span>How to Use CineScope</span>
                </h4>
                <button
                  onClick={() => setShowInfo(false)}
                  className="text-slate-400 hover:text-white transition-colors text-xl leading-none"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-3 text-white/90 text-xs sm:text-sm">
                <div>
                  <p className="font-semibold text-green-400 mb-1">
                    📝 For Movie Reviews:
                  </p>
                  <p className="text-slate-300 leading-relaxed">
                    Enter multiple reviews or a single detailed review to get
                    sentiment analysis like "Masterpiece", "Worth Watching",
                    "Waste of Time", etc.
                  </p>
                </div>

                <div>
                  <p className="font-semibold text-blue-400 mb-1">
                    🎬 For Movie Search (use # prefix):
                  </p>
                  <div className="space-y-1 text-slate-300">
                    <p className="mb-2">
                      Get detailed movie info and related reviews:
                    </p>
                    <div className="bg-slate-700/50 rounded-lg p-2 space-y-1 font-mono text-xs">
                      <p>
                        <span className="text-cyan-400">#</span>The Matrix
                      </p>
                      <p>
                        <span className="text-cyan-400">#</span>Inception
                        Director Christopher Nolan
                      </p>
                      <p>
                        <span className="text-cyan-400">#</span>Avengers 2019
                        Marvel
                      </p>
                      <p>
                        <span className="text-cyan-400">#</span>Parasite Korean
                        Bong Joon-ho
                      </p>
                      <p>
                        <span className="text-cyan-400">#</span>RRR Telugu 2022
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <p className="font-semibold text-purple-400 mb-1">
                    🎯 Include Specifics:
                  </p>
                  <p className="text-slate-300 text-xs leading-relaxed">
                    Add{" "}
                    <span className="text-yellow-400 font-medium">
                      director, cast, year, language, genre
                    </span>{" "}
                    for better results
                  </p>
                </div>

                <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-lg p-2 border border-green-500/30">
                  <p className="text-green-300 font-semibold text-xs flex items-center space-x-1">
                    <span>✨</span>
                    <span>Pro Tip:</span>
                  </p>
                  <p className="text-green-200 text-xs mt-1">
                    More details = Better analysis and movie recommendations!
                  </p>
                </div>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Floating Background Elements - Responsive positioning */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-10 sm:top-20 left-5 sm:left-10 w-16 sm:w-24 lg:w-32 h-16 sm:h-24 lg:h-32 bg-gradient-to-r from-blue-400/20 to-purple-400/20 rounded-full animate-float blur-xl"></div>
        <div
          className="absolute top-20 sm:top-40 right-10 sm:right-20 w-12 sm:w-16 lg:w-24 h-12 sm:h-16 lg:h-24 bg-gradient-to-r from-pink-400/20 to-red-400/20 rounded-full animate-float blur-xl"
          style={{ animationDelay: "2s" }}
        ></div>
        <div
          className="absolute bottom-20 sm:bottom-40 left-10 sm:left-20 w-10 sm:w-14 lg:w-20 h-10 sm:h-14 lg:h-20 bg-gradient-to-r from-green-400/20 to-teal-400/20 rounded-full animate-float blur-xl"
          style={{ animationDelay: "4s" }}
        ></div>
        <div
          className="absolute bottom-10 sm:bottom-20 right-20 sm:right-40 w-14 sm:w-20 lg:w-28 h-14 sm:h-20 lg:h-28 bg-gradient-to-r from-yellow-400/20 to-orange-400/20 rounded-full animate-float blur-xl"
          style={{ animationDelay: "1s" }}
        ></div>
      </div>

      <div className="relative z-10 min-h-screen flex flex-col items-center justify-center p-2 sm:p-4 lg:p-6">
        {/* Main Container - Responsive padding and width */}
        <div className="glass-effect rounded-2xl sm:rounded-3xl p-4 sm:p-6 lg:p-8 w-full max-w-sm sm:max-w-2xl lg:max-w-4xl xl:max-w-5xl space-y-6 sm:space-y-8 transform transition-all duration-500 hover:scale-[1.01] sm:hover:scale-[1.02] animate-fadeInScale shadow-2xl">
          {/* Header Section - Responsive typography and spacing */}
          <div className="text-center space-y-3 sm:space-y-4 animate-fadeInUp">
            <div className="inline-flex items-center justify-center w-16 sm:w-20 h-16 sm:h-20 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl sm:rounded-2xl mb-3 sm:mb-4 animate-pulse-custom shadow-lg">
              <span className="text-2xl sm:text-3xl">🎬</span>
            </div>
            <h1 className="text-3xl sm:text-4xl lg:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-3 sm:mb-4 leading-tight">
              CineScope Analyzer
            </h1>
            <p className="text-base sm:text-lg lg:text-xl text-white/80 max-w-xl lg:max-w-2xl mx-auto leading-relaxed px-2">
              Harness the power of AI to analyze movie reviews and discover
              cinematic insights
            </p>
            <div className="w-16 sm:w-24 h-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full mx-auto"></div>
          </div>

          {/* Input Section - Responsive layout */}
          <div
            className="space-y-4 sm:space-y-6 animate-fadeInUp"
            style={{ animationDelay: "0.2s" }}
          >
            <div className="relative">
              <div className="flex items-center space-x-2 mb-2 sm:mb-3 px-2">
                <label className="text-white/90 text-base sm:text-lg font-semibold">
                  Enter a review to analyze its sentiment or search movies using #Title (e.g., #Inception).
                </label>

                {/* Info Button */}
                <div className="relative inline-block">
                  <button
                    className="w-6 sm:w-7 h-6 sm:h-7 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full flex items-center justify-center text-white text-xs sm:text-sm font-bold hover:from-cyan-600 hover:to-blue-600 transition-all duration-300 shadow-lg hover:shadow-xl transform hover:scale-110 active:scale-95 focus:outline-none focus:ring-2 focus:ring-cyan-400/50"
                    onClick={() => setShowInfo(!showInfo)} // Toggle on click
                    title="Click for detailed instructions"
                  >
                    i
                  </button>
                </div>
              </div>

              <div className="relative">
                <textarea
                  className="w-full h-32 sm:h-40 p-4 sm:p-6 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl sm:rounded-2xl focus:ring-2 focus:ring-blue-500/50 focus:border-blue-400 text-white text-base sm:text-lg resize-none transition-all duration-300 placeholder-white/50 shadow-inner"
                  placeholder="e.g., 'This movie was absolutely brilliant, a true cinematic gem!' or '#TheMatrix' or '#Inception Director Christopher Nolan'"
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  disabled={loading || loadingExample}
                />
                {loading && (
                  <div className="absolute inset-0 bg-white/5 rounded-xl sm:rounded-2xl flex items-center justify-center backdrop-blur-sm">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-6 sm:h-8 w-6 sm:w-8 border-t-2 border-blue-400"></div>
                      <span className="text-white/80 font-medium text-sm sm:text-base"></span>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Action Buttons - Responsive layout */}
            <div className="flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
              <button
                onClick={handleSubmit}
                className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 sm:py-4 rounded-xl sm:rounded-2xl text-lg sm:text-xl font-bold hover:from-blue-700 hover:to-purple-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
                disabled={loading || loadingExample}
              >
                {loading ? (
                  <span className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-4 sm:h-5 w-4 sm:w-5 border-t-2 border-white"></div>
                    <span>Analyzing...</span>
                  </span>
                ) : (
                  "🔍 Analyze Sentiment"
                )}
              </button>
              <button
                onClick={handleTryItOut}
                className="sm:flex-none bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 sm:py-4 px-6 sm:px-8 rounded-xl sm:rounded-2xl text-lg sm:text-xl font-bold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
                disabled={loading || loadingExample}
              >
                {loadingExample ? (
                  <span className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-4 sm:h-5 w-4 sm:w-5 border-t-2 border-white"></div>
                    <span>Loading...</span>
                  </span>
                ) : (
                  "✨ Try Example"
                )}
              </button>
            </div>
          </div>

          {/* Error Display - Responsive padding */}
          {error && (
            <div className="animate-fadeInScale">
              <div className="bg-gradient-to-r from-red-500/20 to-pink-500/20 backdrop-blur-sm border border-red-400/30 text-red-200 p-3 sm:p-4 rounded-xl sm:rounded-2xl relative overflow-hidden">
                <div className="relative z-10">
                  <div className="flex items-start sm:items-center space-x-3">
                    <span className="text-xl sm:text-2xl flex-shrink-0">
                      ⚠️
                    </span>
                    <span className="font-medium text-sm sm:text-base leading-relaxed">
                      {error}
                    </span>
                  </div>
                </div>
                <div className="absolute inset-0 bg-gradient-to-r from-red-600/10 to-transparent"></div>
              </div>
            </div>
          )}

          {/* Results Section - Responsive layout */}
          {(frontendClassification || descriptivePassage) && (
            <div className="animate-fadeInScale space-y-4 sm:space-y-6">
              <div className="glass-effect rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 shadow-xl">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6">
                  <div
                    className={`w-10 sm:w-12 h-10 sm:h-12 bg-gradient-to-r ${getSentimentGradient(
                      frontendClassification
                    )} rounded-lg sm:rounded-xl flex items-center justify-center shadow-lg animate-pulse-custom`}
                  >
                    <span className="text-lg sm:text-xl">🎯</span>
                  </div>
                  <div>
                    <h2 className="text-xl sm:text-2xl font-bold text-white mb-1">
                      Sentiment Analysis
                    </h2>
                    <div className="w-12 sm:w-16 h-1 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"></div>
                  </div>
                </div>

                <div className="space-y-3 sm:space-y-4">
                  <div
                    className={`text-xl sm:text-2xl lg:text-3xl font-black ${getSentimentColor(
                      frontendClassification
                    )} flex items-center space-x-2 sm:space-x-3 flex-wrap`}
                  >
                    <span>🏆</span>
                    <span className="break-words">
                      {frontendClassification}
                    </span>{" "}
                    {/* Display the mapped classification */}
                  </div>
                  <p className="text-white/90 text-base sm:text-lg leading-relaxed font-medium">
                    {descriptivePassage}
                  </p>
                  <div className="flex flex-col sm:flex-row sm:items-center space-y-2 sm:space-y-0 sm:space-x-2 text-xs sm:text-sm text-white/60 mt-3 sm:mt-4 p-3 sm:p-4 bg-white/5 rounded-lg sm:rounded-xl">
                    <span className="text-base sm:text-lg">🤖</span>
                    <span className="font-medium">AI Analysis:</span>
                    <span>
                      Powered by neural networks trained on IMDb reviews
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <MovieDetailsCard movieDetails={movieDetails} />
          <MovieReviewsCard movieReviews={movieReviews} />
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return <MovieAnalyzer />;
}