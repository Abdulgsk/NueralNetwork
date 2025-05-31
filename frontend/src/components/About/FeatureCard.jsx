// src/components/FeatureCard.js
import React from 'react';

function FeatureCard({ icon, title, description, details, gradientFrom, gradientTo, textColor }) {
  const iconBgGradient = `bg-gradient-to-br from-${gradientFrom} via-${gradientFrom.replace(/-\d+/, '-600')} to-${gradientTo}`;
  const hoverBgGradient = `from-${gradientFrom.replace(/-\d+/, '-50')}/50 to-${gradientTo.replace(/-\d+/, '-50')}/50`;
  const detailBorderColor = `border-${gradientFrom}`;
  const detailTextColor = `text-${gradientFrom.replace(/-\d+/, '-700')}`;
  const descriptionHighlightColor = `text-${textColor}`;

  return (
    <div className="group bg-white/90 backdrop-blur-sm rounded-2xl sm:rounded-3xl p-6 sm:p-8 md:p-10 shadow-xl hover:shadow-2xl transition-all duration-500 transform hover:-translate-y-2 border border-white/20 relative overflow-hidden">
      <div className={`absolute inset-0 bg-gradient-to-br ${hoverBgGradient} opacity-0 group-hover:opacity-100 transition-opacity duration-500`}></div>
      
      <div className="relative z-10">
        <div className="flex flex-col sm:flex-row items-start sm:items-center mb-4 sm:mb-6">
          <div className={`w-12 h-12 sm:w-14 sm:h-14 md:w-16 md:h-16 ${iconBgGradient} rounded-xl sm:rounded-2xl flex items-center justify-center text-white text-xl sm:text-2xl font-bold mb-3 sm:mb-0 sm:mr-4 md:mr-6 shadow-lg transform group-hover:rotate-6 transition-transform duration-300 flex-shrink-0`}>
            {icon}
          </div>
          <h3 className="text-xl sm:text-2xl md:text-3xl font-bold bg-gradient-to-r from-gray-800 to-gray-700 bg-clip-text text-transparent leading-tight">
            {title}
          </h3>
        </div>
        
        <p className="text-gray-700 leading-relaxed text-sm sm:text-base md:text-lg mb-4">
          <span className={`font-semibold ${descriptionHighlightColor}`}>{description.highlight}</span> {description.rest}
        </p>
        
        {details && (
          <div className={`bg-gradient-to-r from-gray-50 to-blue-50 rounded-xl sm:rounded-2xl p-4 sm:p-6 border-l-4 ${detailBorderColor}`}>
            <h4 className="font-semibold text-gray-800 mb-3 text-base sm:text-lg">Key Implementations:</h4>
            <div className="space-y-2 sm:space-y-3 text-gray-700 text-sm sm:text-base">
              {details.map((item, index) => (
                <div key={index} className="flex items-start">
                  <span className={`inline-block w-2 h-2 ${item.bulletColor} rounded-full mt-2 mr-3 flex-shrink-0`}></span>
                  <span><strong className={item.strongColor}>{item.strongText}:</strong> {item.plainText}</span>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {details && (
          <p className="text-gray-600 mt-4 font-medium text-sm sm:text-base">
            Trained on thousands of IMDb reviews for accurate sentiment analysis without pre-built frameworks.
          </p>
        )}
      </div>
    </div>
  );
}

export default FeatureCard;