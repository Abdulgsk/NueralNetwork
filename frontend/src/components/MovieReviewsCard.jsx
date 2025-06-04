import React, { useState } from 'react';

const MovieReviewsCard = ({ movieReviews }) => {
  const [expandedReviews, setExpandedReviews] = useState(new Set());

  if (!movieReviews || movieReviews.length === 0) {
    return null;
  }

  // Helper function to get sentiment color and icon
  const getSentimentData = (sentiment) => {
    switch (sentiment.toLowerCase()) {
      case 'positive':
        return {
          textColor: 'text-green-400',
          bgColor: 'bg-green-500/20',
          borderColor: 'border-green-500/30',
          icon: '😊'
        };
      case 'negative':
        return {
          textColor: 'text-red-400',
          bgColor: 'bg-red-500/20',
          borderColor: 'border-red-500/30',
          icon: '😞'
        };
      case 'neutral':
        return {
          textColor: 'text-blue-400',
          bgColor: 'bg-blue-500/20',
          borderColor: 'border-blue-500/30',
          icon: '😐'
        };
      default:
        return {
          textColor: 'text-gray-400',
          bgColor: 'bg-gray-500/20',
          borderColor: 'border-gray-500/30',
          icon: '🤔'
        };
    }
  };

  // Helper function to get star rating display
  const renderStars = (rating) => {
    if (!rating) return null;
    
    const numRating = parseFloat(rating);
    const maxStars = rating.includes('/10') ? 10 : rating.includes('/5') ? 5 : 10;
    const normalizedRating = (numRating / maxStars) * 5;
    const fullStars = Math.floor(normalizedRating);
    const hasHalfStar = normalizedRating % 1 >= 0.5;
    
    return (
      <div className="flex items-center space-x-1">
        {[...Array(5)].map((_, i) => (
          <span
            key={i}
            className={`text-sm ${
              i < fullStars
                ? 'text-yellow-400'
                : i === fullStars && hasHalfStar
                ? 'text-yellow-400'
                : 'text-gray-600'
            }`}
          >
            {i < fullStars ? '★' : i === fullStars && hasHalfStar ? '⭐' : '☆'}
          </span>
        ))}
        <span className="text-xs text-yellow-300 ml-1 font-medium">
          {rating}
        </span>
      </div>
    );
  };

  // Helper function to truncate text
  const truncateText = (text, maxLength = 150) => {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
  };

  // Toggle expanded state for a review
  const toggleExpanded = (index) => {
    const newExpanded = new Set(expandedReviews);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedReviews(newExpanded);
  };

  return (
    <div className="animate-fadeInScale">
      <div className="glass-effect rounded-xl sm:rounded-2xl p-4 sm:p-6 lg:p-8 shadow-xl">
        {/* Header Section */}
        <div className="flex items-center justify-between mb-6 sm:mb-8">
          <div className="flex items-center space-x-3 sm:space-x-4">
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
          <div className="flex items-center space-x-2 px-3 py-1 bg-purple-500/20 rounded-full border border-purple-500/30">
            <span className="text-purple-300 text-sm font-medium">
              {movieReviews.length} Reviews
            </span>
          </div>
        </div>

        {/* Reviews Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 sm:gap-6 mb-2 ">
          {movieReviews.map((review, index) => {
            const sentimentData = getSentimentData(review.sentiment || '');
            const isExpanded = expandedReviews.has(index);
            const shouldTruncate = review.review_text && review.review_text.length > 150;
            
            return (
              <div
                key={index}
                className="group bg-gradient-to-br from-white/5 to-white/10 rounded-xl p-4 sm:p-5 border border-purple-500/20 hover:border-purple-400/40 transition-all duration-300 hover:shadow-lg hover:shadow-purple-500/10 hover:scale-[1.02] flex flex-col h-full"
              >
                {/* Review Text */}
                <div className="flex-1 mb-4 ">
                  <div className="relative">
                    <div className="absolute -top-2 -left-2 text-purple-400/30 text-4xl font-serif">"</div>
                    <p className="text-white/90 text-sm sm:text-base leading-relaxed pl-4 italic">
                      {shouldTruncate && !isExpanded
                        ? truncateText(review.review_text)
                        : review.review_text}
                    </p>
                    <div className="absolute -bottom-2 -right-2 text-purple-400/30 text-4xl font-serif rotate-180">"</div>
                  </div>
                  
                  {/* Expand/Collapse Button */}
                  {shouldTruncate && (
                    <button
                      onClick={() => toggleExpanded(index)}
                      className="text-purple-400 hover:text-purple-300 text-xs mt-2 transition-colors duration-200 flex items-center space-x-1"
                    >
                      <span>{isExpanded ? 'Show less' : 'Read more'}</span>
                      <span className={`transition-transform duration-200 ${isExpanded ? 'rotate-180' : ''}`}>
                        ▼
                      </span>
                    </button>
                  )}
                </div>

                {/* Rating and Sentiment */}
                <div className="flex items-center justify-between flex-wrap gap-2 border-t border-white/10">
                  {/* Rating */}
                  <div className="flex items-center">
                    {review.rating && renderStars(review.rating)}
                  </div>

                  {/* Sentiment Badge */}
                  {review.sentiment && (
                    <div className={`px-2 py-1 rounded-full text-xs font-medium border ${sentimentData.bgColor} ${sentimentData.borderColor} transition-all duration-300 group-hover:scale-105`}>
                      <span className={`flex items-center space-x-1 ${sentimentData.textColor}`}>
                        <span>{sentimentData.icon}</span>
                        <span>{review.sentiment.charAt(0).toUpperCase() + review.sentiment.slice(1)}</span>
                      </span>
                    </div>
                  )}
                </div>

                {/* Helpful/Like Count */}
                {review.helpful_count && (
                  <div className="mt-3 pt-3 border-t border-white/10">
                    <div className="flex items-center space-x-1 text-xs text-white/60">
                      <span>👍</span>
                      <span>{review.helpful_count} found this helpful</span>
                    </div>
                  </div>
                )}

                {/* Review Source */}
                {review.source && (
                  <div className="mt-2">
                    <span className="text-xs text-purple-400/80 bg-purple-500/10 px-2 py-0.5 rounded-full">
                      {review.source}
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        <div className="mb-2 p-2 border-b border-white/10" />
      </div>
    </div>
  );
};

export default MovieReviewsCard;