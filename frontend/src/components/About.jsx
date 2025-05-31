// src/components/AboutSection.js
import React from 'react';

function AboutSection() {
  return (
    <section id="about" className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-20 pt-24 md:pt-32"> {/* Added padding top to clear fixed navbar */}
      <div className="container mx-auto px-8">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-black text-gray-800 mb-6 animate-fadeInUp">
            About CineScope
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto animate-fadeInUp" style={{animationDelay: '0.2s'}}>
            Discover how our AI-powered neural network transforms movie review analysis
          </p>
        </div>
        <div className="grid md:grid-cols-2 gap-16 items-center">
          <div className="space-y-8 animate-fadeInLeft">
            <div className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-xl font-bold mr-4">
                  🧠
                </div>
                <h3 className="text-2xl font-bold text-gray-800">Neural Network Magic</h3>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Our sophisticated neural network, trained on thousands of IMDb reviews, understands the nuances of human sentiment and provides accurate movie ratings.
              </p>
            </div>
            <div className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-teal-600 rounded-full flex items-center justify-center text-white text-xl font-bold mr-4">
                  ⚡
                </div>
                <h3 className="text-2xl font-bold text-gray-800">Lightning Fast Analysis</h3>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Get instant sentiment analysis and descriptive insights from any movie review, helping you make informed viewing decisions.
              </p>
            </div>
            <div className="bg-white rounded-2xl p-8 shadow-xl hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2">
              <div className="flex items-center mb-4">
                <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center text-white text-xl font-bold mr-4">
                  🎯
                </div>
                <h3 className="text-2xl font-bold text-gray-800">Smart Recommendations</h3>
              </div>
              <p className="text-gray-600 leading-relaxed">
                Discover new movies based on your preferences and get detailed information about films you're curious about.
              </p>
            </div>
          </div>
          <div className="animate-fadeInUp">
            <div className="bg-gradient-to-br from-blue-600 to-purple-700 rounded-3xl p-12 text-white shadow-2xl transform hover:scale-105 transition-all duration-300">
              <h3 className="text-3xl font-bold mb-6">How It Works</h3>
              <div className="space-y-6">
                <div className="flex items-start">
                  <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-4 mt-1">
                    <span className="text-sm font-bold">1</span>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Input Analysis</h4>
                    <p className="text-white/80">Enter your movie review or search for a specific film</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-4 mt-1">
                    <span className="text-sm font-bold">2</span>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">AI Processing</h4>
                    <p className="text-white/80">Our neural network analyzes sentiment and context</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center mr-4 mt-1">
                    <span className="text-sm font-bold">3</span>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2">Smart Results</h4>
                    <p className="text-white/80">Get detailed insights and personalized recommendations</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default AboutSection;