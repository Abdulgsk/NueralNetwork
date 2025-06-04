import React from 'react';

const MovieDetailsCard = ({ movieDetails }) => {
  if (!movieDetails || !movieDetails.title) {
    return null; // Don't render if no movie details are available
  }

  // Handle both uppercase and lowercase property names
  const ratings = movieDetails.Ratings || movieDetails.ratings;
  const metascore = movieDetails.Metascore || movieDetails.metascore;
  const imdbVotes = movieDetails.imdbVotes || movieDetails.imdb_votes;

  return (
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

        {/* Hero Section with Poster and Title */}
        <div className="flex flex-col lg:flex-row gap-6 lg:gap-8 mb-8">
          {/* Poster */}
          <div className="flex justify-center lg:justify-start">
            {movieDetails.poster_url ? (
              <div className="relative group">
                <img
                  src={movieDetails.poster_url}
                  alt={`${movieDetails.title} Poster`}
                  className="rounded-2xl shadow-2xl w-48 sm:w-56 lg:w-64 h-auto border-2 border-white/20 transition-all duration-300 group-hover:scale-105 group-hover:shadow-3xl"
                  onError={(e) => {
                    e.target.onerror = null;
                    e.target.src = `https://placehold.co/300x450/334155/E2E8F0?text=No+Poster`;
                  }}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-2xl"></div>
              </div>
            ) : (
              <div className="w-48 sm:w-56 lg:w-64 h-72 sm:h-80 lg:h-96 bg-gradient-to-br from-slate-700 to-slate-800 rounded-2xl flex items-center justify-center border-2 border-white/20">
                <span className="text-4xl">🎬</span>
              </div>
            )}
          </div>

          {/* Details Section */}
          <div className="flex-1 space-y-4 lg:space-y-6">
            {/* Title and Basic Info */}
            <div>
              <h4 className="text-3xl sm:text-4xl lg:text-5xl font-black text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 mb-3 leading-tight">
                {movieDetails.title}
              </h4>
              <div className="flex flex-wrap items-center gap-3 mb-4">
                {movieDetails.year && (
                  <span className="px-3 py-1 bg-blue-500/20 text-blue-300 rounded-full text-sm font-medium border border-blue-500/30">
                    📅 {movieDetails.year}
                  </span>
                )}
                {movieDetails.rated && (
                  <span className="px-3 py-1 bg-green-500/20 text-green-300 rounded-full text-sm font-medium border border-green-500/30">
                    🎯 {movieDetails.rated}
                  </span>
                )}
                {movieDetails.runtime_minutes && (
                  <span className="px-3 py-1 bg-purple-500/20 text-purple-300 rounded-full text-sm font-medium border border-purple-500/30">
                    ⏱️ {movieDetails.runtime_minutes} min
                  </span>
                )}
                {movieDetails.type && (
                  <span className="px-3 py-1 bg-pink-500/20 text-pink-300 rounded-full text-sm font-medium border border-pink-500/30 capitalize">
                    🎭 {movieDetails.type}
                  </span>
                )}
              </div>
            </div>

            {/* Enhanced Ratings Section */}
            {ratings && ratings.length > 0 && (
              <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 rounded-2xl p-4 lg:p-6 border border-yellow-500/20">
                <h5 className="text-xl font-bold text-yellow-300 mb-4 flex items-center">
                  <span className="mr-2">⭐</span>
                  Ratings & Reviews
                </h5>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                  {ratings.map((rating, index) => (
                    <div
                      key={index}
                      className="text-center bg-white/5 rounded-xl p-3 border border-yellow-500/30 hover:bg-white/10 transition-all duration-300"
                    >
                      <div className="text-2xl font-black text-yellow-100 mb-1">
                        {rating.Value || rating.value}
                      </div>
                      <div className="text-xs text-yellow-200 font-medium">
                        {rating.Source || rating.source}
                      </div>
                    </div>
                  ))}
                </div>

                {/* IMDb Votes */}
                {imdbVotes && (
                  <div className="text-center bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-xl p-3 border border-blue-500/30">
                    <div className="text-lg font-bold text-blue-300">
                      {imdbVotes} IMDb Votes
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Box Office Info */}
            {movieDetails.box_office && (
              <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 rounded-2xl p-4 lg:p-6 border border-green-500/20">
                <h5 className="text-xl font-bold text-green-400 mb-3 flex items-center">
                  <span className="mr-2">💰</span>
                  Box Office
                </h5>
                <div className="text-2xl font-black text-green-300">
                  ${movieDetails.box_office.toLocaleString()}
                </div>
                <div className="text-sm text-green-200 mt-1">
                  Worldwide Gross
                </div>
              </div>
            )}

            {/* Metascore */}
            {metascore && (
              <div className="bg-gradient-to-r from-purple-500/10 to-indigo-500/10 rounded-2xl p-4 border border-purple-500/20">
                <h5 className="text-lg font-bold text-purple-400 mb-2 flex items-center">
                  <span className="mr-2">📊</span>
                  Metascore
                </h5>
                <div className="text-3xl font-black text-purple-300">
                  {metascore}/100
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Additional Information Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8">
          {/* Production Details */}
          <div className="space-y-4">
            <h5 className="text-xl font-bold text-cyan-400 mb-4 flex items-center">
              <span className="mr-2">🎭</span>
              Production Details
            </h5>

            <div className="space-y-3">
              {/* Director */}
              {movieDetails.director && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-cyan-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">🎬</span>
                    Director
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.director}
                  </div>
                </div>
              )}

              {/* Writer */}
              {movieDetails.writer && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-cyan-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">✍️</span>
                    Writer
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.writer}
                  </div>
                </div>
              )}

              {/* Cast */}
              {movieDetails.actors &&
                movieDetails.actors.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                    <div className="text-cyan-400 font-semibold text-sm mb-1 flex items-center">
                      <span className="mr-2">👥</span>
                      Main Cast
                    </div>
                    <div className="text-white font-medium">
                      {Array.isArray(movieDetails.actors)
                        ? movieDetails.actors.join(", ")
                        : movieDetails.actors}
                    </div>
                  </div>
                )}

              {/* Genre */}
              {movieDetails.genres &&
                movieDetails.genres.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                    <div className="text-cyan-400 font-semibold text-sm mb-1 flex items-center">
                      <span className="mr-2">🎪</span>
                      Genres
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {(Array.isArray(movieDetails.genres)
                        ? movieDetails.genres
                        : movieDetails.genres.split(", ")
                      ).map((genre, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-cyan-500/20 text-cyan-300 rounded-lg text-xs border border-cyan-500/30"
                        >
                          {genre.trim()}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

              {/* Production Company */}
              {movieDetails.production && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-cyan-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">🏢</span>
                    Production
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.production}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Release & Technical Details */}
          <div className="space-y-4">
            <h5 className="text-xl font-bold text-purple-400 mb-4 flex items-center">
              <span className="mr-2">📅</span>
              Release & Details
            </h5>

            <div className="space-y-3">
              {/* Release Date */}
              {movieDetails.released_date && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">🗓️</span>
                    Release Date
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.released_date}
                  </div>
                </div>
              )}

              {/* Countries */}
              {movieDetails.countries &&
                movieDetails.countries.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                    <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                      <span className="mr-2">🌍</span>
                      Country
                    </div>
                    <div className="text-white font-medium">
                      {Array.isArray(movieDetails.countries)
                        ? movieDetails.countries.join(", ")
                        : movieDetails.countries}
                    </div>
                  </div>
                )}

              {/* Languages */}
              {movieDetails.languages &&
                movieDetails.languages.length > 0 && (
                  <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                    <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                      <span className="mr-2">🗣️</span>
                      Languages
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {(Array.isArray(movieDetails.languages)
                        ? movieDetails.languages
                        : movieDetails.languages.split(", ")
                      ).map((language, index) => (
                        <span
                          key={index}
                          className="px-2 py-1 bg-purple-500/20 text-purple-300 rounded-lg text-xs border border-purple-500/30"
                        >
                          {language.trim()}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              {/* Total Seasons (for TV Series) */}
              {movieDetails.total_seasons && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">📺</span>
                    Total Seasons
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.total_seasons}
                  </div>
                </div>
              )}

              {/* DVD Release */}
              {movieDetails.dvd_release && (
                <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                  <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                    <span className="mr-2">💿</span>
                    DVD Release
                  </div>
                  <div className="text-white font-medium">
                    {movieDetails.dvd_release}
                  </div>
                </div>
              )}

              {/* Website */}
              {movieDetails.website &&
                movieDetails.website !== "N/A" && (
                  <div className="bg-white/5 rounded-xl p-3 border border-white/10 hover:bg-white/10 transition-all duration-300">
                    <div className="text-purple-400 font-semibold text-sm mb-1 flex items-center">
                      <span className="mr-2">🌐</span>
                      Official Website
                    </div>
                    <a
                      href={movieDetails.website}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-blue-400 hover:text-blue-300 font-medium transition-colors duration-300 underline"
                    >
                      Visit Website
                    </a>
                  </div>
                )}
            </div>
          </div>
        </div>

        {/* Plot Synopsis */}
        {movieDetails.plot && (
          <div className="bg-gradient-to-r from-indigo-500/10 to-purple-500/10 rounded-2xl p-4 lg:p-6 border border-indigo-500/20 mt-8">
            <h5 className="text-xl font-bold text-indigo-400 mb-4 flex items-center">
              <span className="mr-2">📖</span>
              Plot Synopsis
            </h5>
            <p className="text-white/90 leading-relaxed text-base lg:text-lg">
              {movieDetails.plot}
            </p>
          </div>
        )}

        {/* Awards & Recognition */}
        {movieDetails.awards && movieDetails.awards !== "N/A" && (
          <div className="bg-gradient-to-r from-amber-500/10 to-yellow-500/10 rounded-2xl p-4 lg:p-6 border border-amber-500/20 mt-6">
            <h5 className="text-xl font-bold text-amber-400 mb-4 flex items-center">
              <span className="mr-2">🏆</span>
              Awards & Recognition
            </h5>
            <p className="text-white/90 leading-relaxed text-base lg:text-lg">
              {movieDetails.awards}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default MovieDetailsCard;