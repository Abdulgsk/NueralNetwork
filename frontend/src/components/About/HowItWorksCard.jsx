// src/components/HowItWorksCard.js
import React from 'react';

function HowItWorksCard() {
  const steps = [
    {
      number: 1,
      title: 'Text Vectorization',
      description: 'Your review is processed using vectorize_single_review() with TF-IDF weighting, Counter-based term frequency calculation, and L2 normalization for optimal neural network input',
      titleColor: 'text-blue-100',
      techDetails: ['TF-IDF Vectorization', 'Counter Class', 'L2 Normalization', '15,000 Word Vocabulary']
    },
    {
      number: 2,
      title: 'Neural Network Processing',
      description: 'Our custom neural network uses forward_pass() with Leaky ReLU activation, softmax output layer, and gradient clipping for stable predictions across multiple hidden layers',
      titleColor: 'text-purple-100',
      techDetails: ['Leaky ReLU Activation', 'Softmax Classification', 'Multi-layer Architecture', 'Dropout Regularization']
    },
    {
      number: 3,
      title: 'Sentiment Interpretation',
      description: 'Results are analyzed through interpret_sentiment_with_passage() providing 9-level classification from "Overwhelmingly Negative" to "Overwhelmingly Positive" with confidence scoring',
      titleColor: 'text-pink-100',
      techDetails: ['9-Level Classification', 'Confidence Scoring', 'Probability Analysis', 'Descriptive Passages']
    },
  ];

  return (
    <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-purple-800 rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-10 lg:p-12 text-white shadow-2xl transform hover:scale-105 transition-all duration-500 relative overflow-hidden">
      {/* Background pattern - mobile adjusted */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-0 right-0 w-20 sm:w-32 md:w-40 h-20 sm:h-32 md:h-40 bg-white rounded-full transform translate-x-10 sm:translate-x-16 md:translate-x-20 -translate-y-10 sm:-translate-y-16 md:-translate-y-20"></div>
        <div className="absolute bottom-0 left-0 w-16 sm:w-24 md:w-32 h-16 sm:h-24 md:h-32 bg-white rounded-full transform -translate-x-8 sm:-translate-x-12 md:-translate-x-16 translate-y-8 sm:translate-y-12 md:translate-y-16"></div>
      </div>
      
      {/* Neural network visualization */}
      <div className="absolute top-4 right-4 opacity-20">
        <div className="flex space-x-2">
          <div className="w-2 h-16 bg-white rounded-full animate-pulse"></div>
          <div className="w-2 h-20 bg-white rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
          <div className="w-2 h-12 bg-white rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
          <div className="w-2 h-8 bg-white rounded-full animate-pulse" style={{animationDelay: '0.6s'}}></div>
        </div>
      </div>
      
      <div className="relative z-10">
        <div className="text-center mb-8">
          <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-3">How Our AI Model Works</h3>
          <p className="text-blue-100 text-sm sm:text-base opacity-90">
            Enhanced Neural Network with TF-IDF • 15,000 Word Vocabulary • 9-Level Classification
          </p>
        </div>
        
        <div className="space-y-6 sm:space-y-8">
          {steps.map((step, index) => (
            <div key={step.number} className="group">
              <div className="flex items-start">
                <div className="w-10 h-10 sm:w-12 sm:h-12 bg-white/20 backdrop-blur-sm rounded-xl sm:rounded-2xl flex items-center justify-center mr-4 sm:mr-6 mt-1 shadow-xl group-hover:bg-white/30 transition-all duration-300 border border-white/20 flex-shrink-0">
                  <span className="text-base sm:text-lg font-bold">{step.number}</span>
                </div>
                <div className="flex-1">
                  <h4 className={`font-bold mb-2 sm:mb-3 text-lg sm:text-xl ${step.titleColor}`}>
                    {step.title}
                  </h4>
                  <p className="text-white/90 text-sm sm:text-base md:text-lg leading-relaxed mb-3">
                    {step.description}
                  </p>
                  
                  {/* Technical details */}
                  <div className="flex flex-wrap gap-2">
                    {step.techDetails.map((tech, idx) => (
                      <span 
                        key={idx}
                        className="px-2 py-1 bg-white/10 backdrop-blur-sm rounded-lg text-xs font-medium border border-white/20 group-hover:bg-white/20 transition-all duration-300"
                      >
                        {tech}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
              
              {/* Connection line (except for last item) */}
              {index < steps.length - 1 && (
                <div className="ml-5 sm:ml-6 mt-4 mb-2">
                  <div className="w-0.5 h-6 bg-gradient-to-b from-white/30 to-transparent"></div>
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* Model specifications */}
        <div className="mt-8 pt-6 border-t border-white/20">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-xl font-bold text-blue-100">15K</div>
              <div className="text-xs text-white/80">Vocabulary</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-xl font-bold text-purple-100">9</div>
              <div className="text-xs text-white/80">Classifications</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-xl font-bold text-pink-100">TF-IDF</div>
              <div className="text-xs text-white/80">Weighting</div>
            </div>
            <div className="bg-white/10 backdrop-blur-sm rounded-xl p-3 border border-white/20">
              <div className="text-xl font-bold text-green-100">L2</div>
              <div className="text-xs text-white/80">Normalized</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HowItWorksCard;