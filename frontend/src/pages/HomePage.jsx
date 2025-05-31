// src/pages/HomePage.js
import { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import HeroSection from '../components/Hero';
import AboutSection from '../components/About';
import ContactSection from '../components/Contact';

function HomePage() {
  const [scrollY, setScrollY] = useState(0);

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (sectionId) => {
    const element = document.getElementById(sectionId);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen">
      {/* Navbar receives scrollY and scrollToSection to control its behavior and enable internal navigation */}
      <Navbar scrollToSection={scrollToSection} scrollY={scrollY} />

      {/* Render individual sections */}
      <HeroSection />
      <AboutSection />
      <ContactSection />
    </div>
  );
}

export default HomePage;