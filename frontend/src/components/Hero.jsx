// src/components/HeroSection.js
import React from 'react';
import { Link } from 'react-router-dom'; // Import Link

function HeroSection() {
  return (
    <section id="hero" className="min-h-screen gradient-bg flex items-center justify-center relative overflow-hidden pt-16 md:pt-0"> {/* Added pt-16 for mobile */}
      {/* Floating Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-20 left-10 w-20 h-20 bg-white/10 rounded-full animate-float"></div>
        <div className="absolute top-40 right-20 w-16 h-16 bg-white/10 rounded-full animate-float" style={{animationDelay: '2s'}}></div>
        <div className="absolute bottom-40 left-20 w-12 h-12 bg-white/10 rounded-full animate-float" style={{animationDelay: '4s'}}></div>
        <div className="absolute bottom-20 right-40 w-24 h-24 bg-white/10 rounded-full animate-float" style={{animationDelay: '1s'}}></div>
      </div>
      <div className="container mx-auto px-8 relative z-10">
        <div className="flex flex-col md:flex-row items-center justify-between">
          {/* Left side content */}
          <div className="flex-1 max-w-2xl text-center md:text-left mb-10 md:mb-0">
            <h1 className="text-5xl md:text-7xl font-black text-white mb-6 animate-fadeInLeft leading-tight"> {/* Adjusted text size for mobile */}
              Welcome to
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
                CineScope
              </span>
            </h1>
            <p className="text-lg md:text-2xl text-white/90 mb-12 animate-fadeInLeft font-light leading-relaxed" style={{animationDelay: '0.2s'}}> {/* Adjusted text size for mobile */}
              Your AI-powered companion for discovering cinematic masterpieces and analyzing movie sentiments with cutting-edge technology.
            </p>
            <Link
              to="/analyze" // Link to the Analyzer page
              className="inline-block bg-black text-white py-3 px-8 md:py-4 md:px-12 rounded-full text-lg md:text-xl font-bold hover:bg-gray-800 transition-all duration-300 ease-in-out shadow-2xl hover:shadow-3xl transform hover:scale-105 active:scale-95 animate-fadeInLeft cursor-pointer" // Adjusted padding and text size for mobile
              style={{animationDelay: '0.4s'}}
            >
              Try It Now
            </Link>
          </div>
          {/* Right side decorative element */}
          <div className="flex-1 flex justify-center items-center">
            <div className="relative">
              <div className="w-60 h-60 md:w-80 md:h-80 bg-gradient-to-br from-white/20 to-white/5 rounded-full animate-pulse-custom backdrop-blur-sm border border-white/30"></div> {/* Adjusted size for mobile */}
              <div className="absolute inset-6 md:inset-8 bg-gradient-to-br from-white/30 to-white/10 rounded-full animate-float backdrop-blur-sm"></div> {/* Adjusted inset for mobile */}
              <div className="absolute inset-12 md:inset-16 bg-gradient-to-br from-white/40 to-white/15 rounded-full animate-pulse-custom backdrop-blur-sm" style={{animationDelay: '1s'}}></div> {/* Adjusted inset for mobile */}
              <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-4xl md:text-6xl">🎬</div> {/* Adjusted icon size for mobile */}
            </div>
          </div>
        </div>
      </div>
      {/* Scroll indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <div className="w-6 h-10 border-2 border-white/50 rounded-full flex justify-center">
          <div className="w-1 h-3 bg-white/70 rounded-full mt-2 animate-pulse"></div>
        </div>
      </div>
    </section>
  );
}

export default HeroSection;