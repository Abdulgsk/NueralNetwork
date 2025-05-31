// src/components/HowItWorksCard.js
import React from 'react';

function HowItWorksCard() {
  const steps = [
    {
      number: 1,
      title: 'Input Analysis',
      description: 'Enter your movie review or search for a specific film to begin the analysis process',
      titleColor: 'text-blue-100'
    },
    {
      number: 2,
      title: 'AI Processing',
      description: 'Our custom neural network analyzes sentiment, context, and linguistic patterns',
      titleColor: 'text-purple-100'
    },
    {
      number: 3,
      title: 'Smart Results',
      description: 'Get detailed insights and personalized recommendations tailored to your preferences',
      titleColor: 'text-pink-100'
    },
  ];

  return (
    <div className="bg-gradient-to-br from-blue-600 via-blue-700 to-purple-800 rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-10 lg:p-12 text-white shadow-2xl transform hover:scale-105 transition-all duration-500 relative overflow-hidden">
      {/* Background pattern - mobile adjusted */}
      <div className="absolute inset-0 opacity-10">
        <div className="absolute top-0 right-0 w-20 sm:w-32 md:w-40 h-20 sm:h-32 md:h-40 bg-white rounded-full transform translate-x-10 sm:translate-x-16 md:translate-x-20 -translate-y-10 sm:-translate-y-16 md:-translate-y-20"></div>
        <div className="absolute bottom-0 left-0 w-16 sm:w-24 md:w-32 h-16 sm:h-24 md:h-32 bg-white rounded-full transform -translate-x-8 sm:-translate-x-12 md:-translate-x-16 translate-y-8 sm:translate-y-12 md:translate-y-16"></div>
      </div>
      
      <div className="relative z-10">
        <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 sm:mb-8 text-center">How It Works</h3>
        <div className="space-y-6 sm:space-y-8">
          {steps.map((step) => (
            <div key={step.number} className="flex items-start group">
              <div className="w-10 h-10 sm:w-12 sm:h-12 bg-white/20 backdrop-blur-sm rounded-xl sm:rounded-2xl flex items-center justify-center mr-4 sm:mr-6 mt-1 shadow-xl group-hover:bg-white/30 transition-all duration-300 border border-white/20 flex-shrink-0">
                <span className="text-base sm:text-lg font-bold">{step.number}</span>
              </div>
              <div className="flex-1">
                <h4 className={`font-bold mb-2 sm:mb-3 text-lg sm:text-xl ${step.titleColor}`}>{step.title}</h4>
                <p className="text-white/90 text-sm sm:text-base md:text-lg leading-relaxed">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default HowItWorksCard;