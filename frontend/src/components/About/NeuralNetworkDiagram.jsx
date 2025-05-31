import React from 'react';

function NeuralNetworkDiagram() {
  const inputNodes = [80, 100, 120, 140, 160];
  const hidden1Nodes = [80, 100, 120, 140, 160, 180, 200];
  const hidden2Nodes = [90, 120, 150, 180, 210];
  const outputNodes = [
    { y: 110, label: '+', color: '#6B7280' },
    { y: 150, label: '−', color: '#6B7280' },
    { y: 190, label: '~', color: '#6B7280' }
  ];

  return (
    <div className="mt-8 lg:mt-12 col-span-full">
      <div className="bg-gradient-to-br from-gray-700 via-gray-600 to-gray-800 backdrop-blur-sm rounded-2xl p-6 md:p-8 shadow-xl border border-gray-500 relative overflow-hidden max-w-2xl mx-auto">
        {/* Subtle background gradient */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-700/90 via-gray-600/80 to-gray-800/90"></div>
        
        <div className="relative z-10">
          <h3 className="text-xl md:text-2xl font-bold text-center mb-6 md:mb-8 bg-gradient-to-r from-gray-100 to-gray-300 bg-clip-text text-transparent">
            Neural Network Architecture
          </h3>
          
          {/* CSS for minimal animations */}
          <style jsx>{`
            @keyframes subtle-pulse {
              0%, 100% { opacity: 0.7; transform: scale(1); }
              50% { opacity: 1; transform: scale(1.05); }
            }
            @keyframes subtle-connection {
              0%, 100% { opacity: 0.3; }
              50% { opacity: 0.5; }
            }
            .animate-subtle-node {
              animation: subtle-pulse 3s ease-in-out infinite;
            }
            .animate-subtle-connection {
              animation: subtle-connection 4s ease-in-out infinite;
            }
          `}</style>
          
          {/* SVG Neural Network */}
          <div className="flex justify-center">
            <svg 
              viewBox="0 0 600 300" 
              className="w-full max-w-lg h-auto"
              style={{ filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.3))' }}
            >
              {/* Connection lines */}
              {/* Input to Hidden Layer 1 */}
              {inputNodes.map((y1, i) => 
                hidden1Nodes.map((y2, j) => (
                  <line
                    key={`ih1-${i}-${j}`}
                    x1="80" y1={y1} x2="200" y2={y2}
                    stroke="url(#gradient1)"
                    strokeWidth="1"
                    opacity="0.3"
                    className="animate-subtle-connection"
                    style={{ animationDelay: `${(i + j) * 0.5}s` }}
                  />
                ))
              )}
              
              {/* Hidden Layer 1 to Hidden Layer 2 */}
              {hidden1Nodes.map((y1, i) => 
                hidden2Nodes.map((y2, j) => (
                  <line
                    key={`h1h2-${i}-${j}`}
                    x1="200" y1={y1} x2="320" y2={y2}
                    stroke="url(#gradient2)"
                    strokeWidth="1"
                    opacity="0.3"
                    className="animate-subtle-connection"
                    style={{ animationDelay: `${(i + j) * 0.6}s` }}
                  />
                ))
              )}
              
              {/* Hidden Layer 2 to Output */}
              {hidden2Nodes.map((y1, i) => 
                outputNodes.map((node, j) => (
                  <line
                    key={`h2o-${i}-${j}`}
                    x1="320" y1={y1} x2="440" y2={node.y}
                    stroke="url(#gradient3)"
                    strokeWidth="1.5"
                    opacity="0.4"
                    className="animate-subtle-connection"
                    style={{ animationDelay: `${(i + j) * 0.7}s` }}
                  />
                ))
              )}

              {/* Gradient definitions */}
              <defs>
                <linearGradient id="gradient1" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#6B7280" />
                  <stop offset="100%" stopColor="#9CA3AF" />
                </linearGradient>
                <linearGradient id="gradient2" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#9CA3AF" />
                  <stop offset="100%" stopColor="#D1D5DB" />
                </linearGradient>
                <linearGradient id="gradient3" x1="0%" y1="0%" x2="100%" y2="0%">
                  <stop offset="0%" stopColor="#D1D5DB" />
                  <stop offset="100%" stopColor="#E5E7EB" />
                </linearGradient>
                <radialGradient id="nodeGradient1" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#9CA3AF" />
                  <stop offset="100%" stopColor="#6B7280" />
                </radialGradient>
                <radialGradient id="nodeGradient2" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#D1D5DB" />
                  <stop offset="100%" stopColor="#9CA3AF" />
                </radialGradient>
                <radialGradient id="nodeGradient3" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#E5E7EB" />
                  <stop offset="100%" stopColor="#D1D5DB" />
                </radialGradient>
                <radialGradient id="nodeGradient4" cx="50%" cy="50%" r="50%">
                  <stop offset="0%" stopColor="#F3F4F6" />
                  <stop offset="100%" stopColor="#E5E7EB" />
                </radialGradient>
              </defs>

              {/* Input Layer Nodes */}
              {inputNodes.map((y, i) => (
                <g key={`input-${i}`}>
                  <circle
                    cx="80" cy={y} r="8"
                    fill="url(#nodeGradient1)"
                    className="animate-subtle-node"
                    style={{ animationDelay: `${i * 1}s` }}
                  />
                  <circle
                    cx="80" cy={y} r="8"
                    fill="none"
                    stroke="#E5E7EB"
                    strokeWidth="1"
                    opacity="0.8"
                  />
                </g>
              ))}

              {/* Hidden Layer 1 Nodes */}
              {hidden1Nodes.map((y, i) => (
                <g key={`hidden1-${i}`}>
                  <circle
                    cx="200" cy={y} r="7"
                    fill="url(#nodeGradient2)"
                    className="animate-subtle-node"
                    style={{ animationDelay: `${i * 1.2}s` }}
                  />
                  <circle
                    cx="200" cy={y} r="7"
                    fill="none"
                    stroke="#9CA3AF"
                    strokeWidth="1"
                    opacity="0.6"
                  />
                </g>
              ))}

              {/* Hidden Layer 2 Nodes */}
              {hidden2Nodes.map((y, i) => (
                <g key={`hidden2-${i}`}>
                  <circle
                    cx="320" cy={y} r="7"
                    fill="url(#nodeGradient3)"
                    className="animate-subtle-node"
                    style={{ animationDelay: `${i * 1.4}s` }}
                  />
                  <circle
                    cx="320" cy={y} r="7"
                    fill="none"
                    stroke="#D1D5DB"
                    strokeWidth="1"
                    opacity="0.7"
                  />
                </g>
              ))}

              {/* Output Layer Nodes */}
              {outputNodes.map((node, i) => (
                <g key={`output-${i}`}>
                  <circle
                    cx="440" cy={node.y} r="10"
                    fill="url(#nodeGradient4)"
                    className="animate-subtle-node"
                    style={{ animationDelay: `${i * 1.8}s` }}
                  />
                  <circle
                    cx="440" cy={node.y} r="10"
                    fill="none"
                    stroke="#9CA3AF"
                    strokeWidth="1"
                    opacity="0.6"
                  />
                  <text
                    x="440" y={node.y + 1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fontSize="10"
                    fontWeight="bold"
                    fill="#374151"
                  >
                    {node.label}
                  </text>
                </g>
              ))}

              {/* Layer Labels */}
              <text x="80" y="40" textAnchor="middle" fontSize="12" fontWeight="600" fill="#F9FAFB">Input</text>
              <text x="80" y="55" textAnchor="middle" fontSize="10" fill="#E5E7EB">Features</text>
              
              <text x="200" y="40" textAnchor="middle" fontSize="12" fontWeight="600" fill="#F9FAFB">Hidden 1</text>
              <text x="200" y="55" textAnchor="middle" fontSize="10" fill="#E5E7EB">Processing</text>
              
              <text x="320" y="40" textAnchor="middle" fontSize="12" fontWeight="600" fill="#F9FAFB">Hidden 2</text>
              <text x="320" y="55" textAnchor="middle" fontSize="10" fill="#E5E7EB">Analysis</text>
              
              <text x="440" y="40" textAnchor="middle" fontSize="12" fontWeight="600" fill="#F9FAFB">Output</text>
              <text x="440" y="55" textAnchor="middle" fontSize="10" fill="#E5E7EB">Sentiment</text>

              {/* Data Flow Arrow */}
              <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                        refX="9" refY="3.5" orient="auto">
                  <polygon points="0 0, 10 3.5, 0 7" fill="#9CA3AF" />
                </marker>
              </defs>
              <line
                x1="120" y1="260" x2="400" y2="260"
                stroke="#9CA3AF"
                strokeWidth="2"
                markerEnd="url(#arrowhead)"
                opacity="0.7"
              />
              <text x="260" y="280" textAnchor="middle" fontSize="11" fill="#D1D5DB" fontWeight="500">
                Information Flow
              </text>
            </svg>
          </div>

          {/* Legend */}
          <div className="flex flex-wrap justify-center gap-4 md:gap-6 mt-6 text-xs md:text-sm text-gray-300">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-gray-700 to-gray-600"></div>
              <span>Text Processing</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-gray-600 to-gray-500"></div>
              <span>Feature Extraction</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-gray-500 to-gray-400"></div>
              <span>Pattern Recognition</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-gradient-to-r from-gray-400 to-gray-300"></div>
              <span>Classification</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default NeuralNetworkDiagram;