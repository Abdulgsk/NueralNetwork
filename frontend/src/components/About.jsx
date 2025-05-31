// src/components/AboutSection.js
import React from 'react';
import AboutHeader from './About/AboutHeader';
import FeatureCard from './About/FeatureCard';
import HowItWorksCard from './About/HowItWorksCard';
import NeuralNetworkDiagram from './About/NeuralNetworkDiagram';

function AboutSection() {
  return (
    <section id="about" className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 py-8 sm:py-12 md:py-16 lg:py-20 pt-16 sm:pt-20 md:pt-24 lg:pt-32 relative overflow-hidden">
      {/* Background decoration - mobile optimized */}
      <div className="absolute inset-0 opacity-5">
        <div className="absolute top-5 left-3 w-20 h-20 sm:w-32 sm:h-32 md:w-48 md:h-48 lg:w-72 lg:h-72 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl animate-pulse"></div>
        <div className="absolute top-10 right-3 w-20 h-20 sm:w-32 sm:h-32 md:w-48 md:h-48 lg:w-72 lg:h-72 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl animate-pulse" style={{animationDelay: '2s'}}></div>
        <div className="absolute bottom-5 left-1/4 w-20 h-20 sm:w-32 sm:h-32 md:w-48 md:h-48 lg:w-72 lg:h-72 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl animate-pulse" style={{animationDelay: '4s'}}></div>
      </div>
      
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Header Section - Mobile First */}
        <AboutHeader />

        {/* Main Content Grid - Mobile Stacked */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 sm:gap-12 md:gap-16 lg:gap-20 items-start">
          
          {/* Features Cards - Mobile Optimized */}
          <div className="space-y-6 sm:space-y-8 animate-fadeInLeft order-2 lg:order-1">
            <FeatureCard
              icon="🧠"
              title="Custom Neural Network Engine"
              description={{ highlight: '', rest: 'Our sentiment analysis is powered by a neural network built entirely from scratch, leveraging NumPy for all mathematical operations.' }}
              details={[
                { bulletColor: 'bg-blue-500', strongColor: 'text-blue-700', strongText: 'Forward Propagation', plainText: 'Processing input reviews through layers with custom activation functions' },
                { bulletColor: 'bg-purple-500', strongColor: 'text-purple-700', strongText: 'Backward Propagation', plainText: 'Calculating gradients for efficient weight updates' },
                { bulletColor: 'bg-green-500', strongColor: 'text-green-700', strongText: 'Optimizers', plainText: 'Fine-tuning network parameters through iterative learning' },
              ]}
              gradientFrom="blue-500"
              gradientTo="purple-600"
              textColor="text-blue-600" // Not directly used for highlight in this specific card, but good to keep consistent
            />

            <FeatureCard
              icon="⚡"
              title="Lightning Fast Analysis"
              description={{ highlight: 'instant sentiment analysis', rest: 'and descriptive insights from any movie review, helping you make informed viewing decisions with unprecedented speed and accuracy.' }}
              gradientFrom="green-500"
              gradientTo="teal-600"
              textColor="green-700"
            />

            <FeatureCard
              icon="🎯"
              title="Smart Recommendations"
              description={{ highlight: 'intelligent recommendation system', rest: 'Discover new movies based on your preferences and get detailed information about films you are curious about.'}}
              gradientFrom="purple-500"
              gradientTo="pink-600"
              textColor="purple-700"
            />
          </div>

          {/* How It Works Section and Neural Network Diagram - Mobile Optimized */}
          {/* By removing the 'order' classes, they will naturally stack in the order they appear */}
          <div className="animate-fadeInUp">
            <HowItWorksCard />
            {/* Neural Network Diagram - Compact Version */}
            <NeuralNetworkDiagram />
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeInUp {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        @keyframes fadeInLeft {
          from {
            opacity: 0;
            transform: translateX(-20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
        
        .animate-fadeInUp {
          animation: fadeInUp 0.8s ease-out forwards;
        }
        
        .animate-fadeInLeft {
          animation: fadeInLeft 0.8s ease-out forwards;
        }
        
        .shadow-3xl {
          box-shadow: 0 35px 60px -12px rgba(0, 0, 0, 0.25);
        }
        
        /* Neural Network Animations */
        @keyframes current-flow {
          0% {
            opacity: 0.3;
            transform: scaleX(0.5);
          }
          50% {
            opacity: 1;
            transform: scaleX(1);
          }
          100% {
            opacity: 0.3;
            transform: scaleX(0.5);
          }
        }
        
        @keyframes pulse-slow {
          0%, 100% {
            opacity: 0.8;
            transform: scale(1);
          }
          50% {
            opacity: 1;
            transform: scale(1.1);
          }
        }
        
        .animate-current-flow {
          animation: current-flow 2s ease-in-out infinite;
        }
        
        .animate-pulse-slow {
          animation: pulse-slow 3s ease-in-out infinite;
        }
        
        /* Enhanced Neural Network Animations */
        @keyframes pulse-connection {
          0%, 100% {
            opacity: 0.2;
            stroke-width: 1;
          }
          50% {
            opacity: 0.6;
            stroke-width: 1.5;
          }
        }
        
        @keyframes pulse-node {
          0%, 100% {
            transform: scale(1);
            opacity: 0.9;
          }
          50% {
            transform: scale(1.1);
            opacity: 1;
          }
        }
        
        @keyframes flow {
          0% {
            stroke-dasharray: 5 5;
            stroke-dashoffset: 0;
          }
          100% {
            stroke-dashoffset: 10;
          }
        }
        
        .animate-pulse-connection {
          animation: pulse-connection 2s ease-in-out infinite;
        }
        
        .animate-pulse-node {
          animation: pulse-node 2.5s ease-in-out infinite;
        }
        
        .animate-flow {
          animation: flow 2s linear infinite;
        }
        
        /* Mobile-specific optimizations */
        @media (max-width: 640px) {
          .container {
            padding-left: 1rem;
            padding-right: 1rem;
          }
        }
      `}</style>
    </section>
  );
}

export default AboutSection;