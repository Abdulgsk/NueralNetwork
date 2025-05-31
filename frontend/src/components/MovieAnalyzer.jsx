import React, { useState } from "react";

function MovieAnalyzer() {
  const [reviewText, setReviewText] = useState("");
  const [sentimentClassification, setSentimentClassification] = useState(null);
  const [descriptivePassage, setDescriptivePassage] = useState(null);
  const [movieDetails, setMovieDetails] = useState(null);
  const [movieReviews, setMovieReviews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setSentimentClassification(null);
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
      const res = await fetch("https://nueralnetwork.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: reviewText }),
      });

      if (!res.ok) {
        const errorData = await res.json();
        throw new Error(
          errorData.error || "Failed to get sentiment from backend."
        );
      }

      const data = await res.json();
      setSentimentClassification(data.sentiment_classification);
      setDescriptivePassage(data.descriptive_passage);

      if (data.details && Object.keys(data.details).length > 0) {
        setMovieDetails(data.details);
      } else {
        setMovieDetails(null);
      }

      if (data.reviews && data.reviews.length > 0) {
        setMovieReviews(data.reviews);
      } else {
        setMovieReviews([]);
      }
    } catch (err) {
      console.error("Error during sentiment analysis:", err);
      setError(`Analysis failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (classification) => {
    if (!classification) return "text-gray-800";
    if (
      classification.includes("Masterpiece") ||
      classification.includes("Must-Watch") ||
      classification.includes("Very Positive") ||
      classification.includes("Highly Recommended") ||
      classification.includes("Generally Positive") ||
      classification.includes("Slightly Positive")
    ) {
      return "text-emerald-600";
    }
    if (
      classification.includes("Worst") ||
      classification.includes("Disappointing") ||
      classification.includes("Negative") ||
      classification.includes("Flop")
    ) {
      return "text-red-500";
    }
    return "text-amber-500";
  };

  const getSentimentGradient = (classification) => {
    if (!classification) return "from-gray-500 to-gray-600";
    if (
      classification.includes("Masterpiece") ||
      classification.includes("Must-Watch") ||
      classification.includes("Very Positive") ||
      classification.includes("Highly Recommended") ||
      classification.includes("Generally Positive") ||
      classification.includes("Slightly Positive")
    ) {
      return "from-emerald-500 to-green-600";
    }
    if (
      classification.includes("Worst") ||
      classification.includes("Disappointing") ||
      classification.includes("Negative") ||
      classification.includes("Flop")
    ) {
      return "from-red-500 to-pink-600";
    }
    return "from-amber-500 to-orange-600";
  };

  const handleTryItOut = async () => {
    try {
      const response = await fetch("https://nueralnetwork-1.onrender.com/fetch_dummy_reviews", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: "some movie" }), // optional here
      });
      const data = await response.json();
      if (data.reviews && data.reviews.length > 0) {
        setReviewText(data.reviews.join(" ")); // or pick one or display as list
      }
    } catch (error) {
      console.error("Failed to fetch dummy movie data:", error);
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
          .animate-float { animation: float 6s ease-in-out infinite; }
          .animate-fadeInUp { animation: fadeInUp 0.6s ease-out; }
          .animate-fadeInScale { animation: fadeInScale 0.5s ease-out; }
          .animate-pulse-custom { animation: pulse-custom 2s ease-in-out infinite; }
          .glass-effect {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.1);
          }
        `}
      </style>

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
              <label className="block text-white/90 text-base sm:text-lg font-semibold mb-2 sm:mb-3 px-2">
                Enter your movie review or search for a movie (Movie Name should
                start with "#").
              </label>
              <div className="relative">
                <textarea
                  className="w-full h-32 sm:h-40 p-4 sm:p-6 bg-white/10 backdrop-blur-sm border border-white/20 rounded-xl sm:rounded-2xl focus:ring-2 focus:ring-blue-500/50 focus:border-blue-400 text-white text-base sm:text-lg resize-none transition-all duration-300 placeholder-white/50 shadow-inner"
                  placeholder="e.g., 'This movie was absolutely brilliant, a true cinematic gem!' or '#TheMatrix'"
                  value={reviewText}
                  onChange={(e) => setReviewText(e.target.value)}
                  disabled={loading}
                />
                {loading && (
                  <div className="absolute inset-0 bg-white/5 rounded-xl sm:rounded-2xl flex items-center justify-center backdrop-blur-sm">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-6 sm:h-8 w-6 sm:w-8 border-t-2 border-blue-400"></div>
                      <span className="text-white/80 font-medium text-sm sm:text-base">
                        Analyzing...
                      </span>
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
                disabled={loading}
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
                className="sm:flex-none bg-gradient-to-r from-purple-600 to-pink-600 text-white py-3 sm:py-4 px-6 sm:px-8 rounded-xl sm:rounded-2xl text-lg sm:text-xl font-bold hover:from-purple-700 hover:to-pink-700 transition-all duration-300 disabled:opacity-50 shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
                disabled={loading}
              >
                ✨ Try Example
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
          {(sentimentClassification || descriptivePassage) && (
            <div className="animate-fadeInScale space-y-4 sm:space-y-6">
              <div className="glass-effect rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 space-y-4 sm:space-y-6 shadow-xl">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6">
                  <div
                    className={`w-10 sm:w-12 h-10 sm:h-12 bg-gradient-to-r ${getSentimentGradient(
                      sentimentClassification
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
                      sentimentClassification
                    )} flex items-center space-x-2 sm:space-x-3 flex-wrap`}
                  >
                    <span>🏆</span>
                    <span className="break-words">
                      {sentimentClassification}
                    </span>
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

          {/* Movie Details - Responsive grid */}
          {movieDetails && movieDetails.title && (
            <div className="animate-fadeInScale">
              <div className="glass-effect rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6">
                  <div className="w-10 sm:w-12 h-10 sm:h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg sm:rounded-xl flex items-center justify-center shadow-lg">
                    <span className="text-lg sm:text-xl">🎭</span>
                  </div>
                  <div>
                    <h3 className="text-xl sm:text-2xl font-bold text-white mb-1">
                      Movie Details
                    </h3>
                    <div className="w-12 sm:w-16 h-1 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"></div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                  <div className="space-y-3 sm:space-y-4">
                    <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3">
                      <span className="text-blue-400 font-bold text-sm sm:text-base">
                        Title:
                      </span>
                      <span className="text-white font-medium text-sm sm:text-base break-words">
                        {movieDetails.title}
                      </span>
                    </div>
                    <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3">
                      <span className="text-blue-400 font-bold text-sm sm:text-base">
                        Year:
                      </span>
                      <span className="text-white font-medium text-sm sm:text-base">
                        {movieDetails.year}
                      </span>
                    </div>
                    <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3">
                      <span className="text-blue-400 font-bold text-sm sm:text-base">
                        Genre:
                      </span>
                      <span className="text-white font-medium text-sm sm:text-base break-words">
                        {movieDetails.genre}
                      </span>
                    </div>
                  </div>
                  <div className="space-y-3 sm:space-y-4">
                    {movieDetails.director && (
                      <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3">
                        <span className="text-blue-400 font-bold text-sm sm:text-base">
                          Director:
                        </span>
                        <span className="text-white font-medium text-sm sm:text-base break-words">
                          {movieDetails.director}
                        </span>
                      </div>
                    )}
                    {movieDetails.cast && (
                      <div className="flex flex-col sm:flex-row sm:items-start space-y-1 sm:space-y-0 sm:space-x-3">
                        <span className="text-blue-400 font-bold text-sm sm:text-base flex-shrink-0">
                          Cast:
                        </span>
                        <span className="text-white font-medium text-sm sm:text-base break-words">
                          {movieDetails.cast}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Related Reviews - Responsive layout */}
          {movieReviews.length > 0 && (
            <div className="animate-fadeInScale">
              <div className="glass-effect rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl">
                <div className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-6">
                  <div className="w-10 sm:w-12 h-10 sm:h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg sm:rounded-xl flex items-center justify-center shadow-lg">
                    <span className="text-lg sm:text-xl">💬</span>
                  </div>
                  <div>
                    <h3 className="text-xl sm:text-2xl font-bold text-white mb-1">
                      Related Reviews
                    </h3>
                    <div className="w-12 sm:w-16 h-1 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full"></div>
                  </div>
                </div>

                <div className="space-y-3 sm:space-y-4">
                  {movieReviews.map((review, index) => (
                    <div
                      key={index}
                      className="bg-white/5 rounded-lg sm:rounded-xl p-3 sm:p-4 border border-white/10 hover:bg-white/10 transition-all duration-300"
                    >
                      <div className="flex items-start space-x-2 sm:space-x-3">
                        <span className="text-purple-400 font-bold text-xs sm:text-sm flex-shrink-0">
                          #{index + 1}
                        </span>
                        <p className="text-white/90 leading-relaxed text-sm sm:text-base break-words">
                          {review}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MovieAnalyzer;
