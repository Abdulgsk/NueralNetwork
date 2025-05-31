// src/components/Navbar.js
import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

function Navbar({ scrollToSection, scrollY }) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  
  // Determine if navbar should have a background based on scroll
  const isScrolled = scrollY > 50;

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen);
  };

  const handleNavClick = (action) => {
    // Close mobile menu when a nav item is clicked
    setIsMobileMenuOpen(false);
    // Execute the action (scroll or navigate)
    if (typeof action === 'function') {
      action();
    }
  };

  return (
    <nav className={`bg-white/10 backdrop-blur-md border-b border-white/20 p-4 fixed w-full z-50 top-0 transition-all duration-300 ${isScrolled ? 'bg-white/20 shadow-lg' : ''}`}>
      <div className="container mx-auto">
        {/* Main navbar content */}
        <div className="flex justify-between items-center">
          {/* Logo/Brand */}
          <button
            onClick={() => handleNavClick(() => scrollToSection('hero'))}
            className="text-white text-2xl md:text-3xl font-black tracking-wider hover:text-blue-200 transition-colors duration-300 cursor-pointer focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-2 py-1"
          >
            CineScope
          </button>

          {/* Desktop Navigation */}
          <div className="hidden md:flex space-x-6 lg:space-x-8">
            <button
              onClick={() => handleNavClick(() => scrollToSection('about'))}
              className="text-white text-lg font-medium hover:text-blue-200 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-3 py-2"
            >
              About
            </button>
            <Link
              to="/analyze"
              className="text-white text-lg font-medium hover:text-blue-200 transition-colors duration-300 focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-3 py-2"
            >
              Analyzer
            </Link>
            <button
              onClick={() => handleNavClick(() => scrollToSection('contact'))}
              className="text-white text-lg font-medium hover:text-blue-200 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-3 py-2"
            >
              Contact
            </button>
          </div>

          {/* Mobile menu button */}
          <button
            onClick={toggleMobileMenu}
            className="md:hidden text-white hover:text-blue-200 transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg p-2"
            aria-label="Toggle navigation menu"
          >
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation Menu */}
        <div className={`md:hidden transition-all duration-300 ease-in-out ${
          isMobileMenuOpen 
            ? 'max-h-64 opacity-100 mt-4' 
            : 'max-h-0 opacity-0 overflow-hidden'
        }`}>
          <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 p-4 space-y-2">
            <button
              onClick={() => handleNavClick(() => scrollToSection('about'))}
              className="block w-full text-left text-white text-lg font-medium hover:text-blue-200 hover:bg-white/10 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-4 py-3"
            >
              About
            </button>
            <Link
              to="/analyze"
              onClick={() => handleNavClick()}
              className="block w-full text-left text-white text-lg font-medium hover:text-blue-200 hover:bg-white/10 transition-all duration-300 focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-4 py-3"
            >
              Analyzer
            </Link>
            <button
              onClick={() => handleNavClick(() => scrollToSection('contact'))}
              className="block w-full text-left text-white text-lg font-medium hover:text-blue-200 hover:bg-white/10 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-50 rounded-lg px-4 py-3"
            >
              Contact
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;