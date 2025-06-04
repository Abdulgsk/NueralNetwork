import React, { useState, useEffect } from 'react';

const AIBrainVisualization = () => {
  const [pulseIndex, setPulseIndex] = useState(0);
  const [floatingElements, setFloatingElements] = useState([]);

  useEffect(() => {
    // Create floating elements
    const elements = Array.from({ length: 12 }, (_, i) => ({
      id: i,
      delay: i * 0.3,
      size: Math.random() * 8 + 4,
      x: Math.random() * 100,
      y: Math.random() * 100,
      duration: Math.random() * 3 + 2,
    }));
    setFloatingElements(elements);

    // Pulse animation cycle
    const interval = setInterval(() => {
      setPulseIndex(prev => (prev + 1) % 3);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  const stats = [
    { label: 'Accuracy', value: '91.2%', color: 'from-emerald-400 to-teal-600' },
    { label: 'Precision', value: '89.7%', color: 'from-blue-400 to-indigo-600' },
    { label: 'Processing', value: '<0.1s', color: 'from-purple-400 to-violet-600' },
  ];

  return (
    <div className="relative bg-gradient-to-br from-slate-900/95 via-blue-900/90 to-purple-900/95 backdrop-blur-sm rounded-2xl p-6 shadow-2xl border border-white/10 hover:shadow-3xl transition-all duration-700 overflow-hidden group">
      
      {/* Animated Background Pattern */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 animate-gradient-x"></div>
        {floatingElements.map((element) => (
          <div
            key={element.id}
            className="absolute w-2 h-2 bg-white/30 rounded-full animate-float"
            style={{
              left: `${element.x}%`,
              top: `${element.y}%`,
              animationDelay: `${element.delay}s`,
              animationDuration: `${element.duration}s`,
            }}
          />
        ))}
      </div>

      {/* Main Content */}
      <div className="relative z-10">
        {/* Header */}
        <div className="text-center mb-6">
          <div className="inline-flex items-center gap-3 mb-3">
            <div className="relative">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-cyan-400 to-blue-600 flex items-center justify-center shadow-lg">
                <span className="text-2xl">🤖</span>
              </div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-400 rounded-full animate-ping"></div>
              <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full"></div>
            </div>
            <div>
              <h3 className="text-xl font-bold text-white">AI Intelligence</h3>
              <p className="text-sm text-blue-200">Neural Processing Unit</p>
            </div>
          </div>
        </div>

        {/* Central Brain Visualization */}
        <div className="flex justify-center mb-6">
          <div className="relative w-32 h-32">
            {/* Brain Core */}
            <div className="absolute inset-0 rounded-full bg-gradient-to-r from-pink-500 via-purple-500 to-indigo-500 animate-spin-slow opacity-80"></div>
            <div className="absolute inset-2 rounded-full bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 animate-pulse"></div>
            <div className="absolute inset-4 rounded-full bg-gradient-to-r from-white via-blue-100 to-purple-100 flex items-center justify-center">
              <span className="text-3xl animate-bounce">🧠</span>
            </div>
            
            {/* Orbiting Elements */}
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className={`absolute w-4 h-4 rounded-full bg-gradient-to-r from-yellow-400 to-orange-500 shadow-lg ${
                  pulseIndex === i ? 'animate-ping' : 'animate-pulse'
                }`}
                style={{
                  top: '50%',
                  left: '50%',
                  transform: `rotate(${i * 120}deg) translateX(60px) translateY(-50%)`,
                  transformOrigin: '0 0',
                }}
              />
            ))}
          </div>
        </div>

        {/* Performance Stats */}
        <div className="grid grid-cols-3 gap-3 mb-6">
          {stats.map((stat, index) => (
            <div
              key={index}
              className="bg-white/10 backdrop-blur-sm rounded-xl p-3 text-center hover:bg-white/20 transition-all duration-300 hover:scale-105"
            >
              <div className={`text-lg font-bold bg-gradient-to-r ${stat.color} bg-clip-text text-transparent`}>
                {stat.value}
              </div>
              <div className="text-xs text-blue-200 mt-1">{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Neural Activity Indicator */}
        <div className="bg-white/10 backdrop-blur-sm rounded-xl p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-sm font-medium text-white">Neural Activity</span>
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
              <span className="text-xs text-green-300">Active</span>
            </div>
          </div>
          
          {/* Activity Bars */}
          <div className="flex items-end gap-1 h-12">
            {Array.from({ length: 12 }).map((_, i) => (
              <div
                key={i}
                className="bg-gradient-to-t from-blue-600 to-cyan-400 rounded-t-sm flex-1 animate-pulse"
                style={{
                  height: `${Math.random() * 80 + 20}%`,
                  animationDelay: `${i * 0.1}s`,
                  animationDuration: `${Math.random() * 2 + 1}s`,
                }}
              />
            ))}
          </div>
        </div>

        {/* Processing Indicator */}
        <div className="mt-4 flex items-center justify-center gap-2">
          <div className="flex gap-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="w-2 h-2 rounded-full bg-cyan-400 animate-bounce"
                style={{ animationDelay: `${i * 0.1}s` }}
              />
            ))}
          </div>
          <span className="text-xs text-cyan-300 ml-2">Processing Reviews...</span>
        </div>
      </div>

      {/* Hover Glow Effect */}
      <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-cyan-500/20 via-blue-500/20 to-purple-500/20 opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"></div>

      <style jsx>{`
        @keyframes float {
          0%, 100% {
            transform: translateY(0px) rotate(0deg);
            opacity: 0.7;
          }
          50% {
            transform: translateY(-20px) rotate(180deg);
            opacity: 1;
          }
        }
        
        @keyframes gradient-x {
          0%, 100% {
            transform: translateX(-100%);
          }
          50% {
            transform: translateX(100%);
          }
        }
        
        @keyframes spin-slow {
          from {
            transform: rotate(0deg);
          }
          to {
            transform: rotate(360deg);
          }
        }
        
        .animate-float {
          animation: float 4s ease-in-out infinite;
        }
        
        .animate-gradient-x {
          animation: gradient-x 6s ease infinite;
        }
        
        .animate-spin-slow {
          animation: spin-slow 8s linear infinite;
        }
        
        .shadow-3xl {
          box-shadow: 0 35px 60px -12px rgba(0, 0, 0, 0.4);
        }
      `}</style>
    </div>
  );
};

export default AIBrainVisualization;