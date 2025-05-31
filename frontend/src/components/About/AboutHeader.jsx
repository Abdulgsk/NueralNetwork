// src/components/AboutHeader.js
import React from 'react';

function AboutHeader() {
  return (
    <div className="text-center mb-8 sm:mb-12 md:mb-16 lg:mb-20">
      <div className="inline-block">
        <h2 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl xl:text-7xl font-black bg-gradient-to-r from-gray-800 via-gray-700 to-gray-900 bg-clip-text text-transparent mb-3 sm:mb-4 md:mb-6 animate-fadeInUp leading-tight tracking-tight px-2">
          About CineScope
        </h2>
        <div className="h-1 w-12 sm:w-16 md:w-20 lg:w-24 bg-gradient-to-r from-blue-500 to-purple-600 mx-auto mb-4 sm:mb-6 md:mb-8 rounded-full animate-fadeInUp" style={{ animationDelay: '0.1s' }}></div>
      </div>
      <p className="text-base sm:text-lg md:text-xl lg:text-2xl text-gray-600 max-w-sm sm:max-w-2xl md:max-w-3xl lg:max-w-4xl mx-auto animate-fadeInUp font-light leading-relaxed px-4" style={{ animationDelay: '0.2s' }}>
        Discover how our <span className="font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">AI-powered neural network</span> transforms movie review analysis with cutting-edge technology
      </p>
    </div>
  );
}

export default AboutHeader;