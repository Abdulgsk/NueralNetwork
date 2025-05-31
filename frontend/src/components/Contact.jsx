// src/components/ContactSection.js
import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import emailjs from 'emailjs-com'; // Ensure this import path is correct for your setup

function ContactSection() {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState('');
  const [isSending, setIsSending] = useState(false); // <--- Re-added: New state for loading indicator

  const CONTACT_EMAIL = 'abdul29112004@gmail.com';
  const LINKEDIN_URL = 'https://www.linkedin.com/in/abdul-gouse-syeedy-000027277';
  const GITHUB_URL = 'https://github.com/Abdulgsk';

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!name || !email || !message) {
      toast.error('Please fill in all fields before sending your message.');
      return;
    }

    setIsSending(true); // <--- Re-added: Set loading to true when sending starts

    const templateParams = {
      from_name: name,
      from_email: email,
      message: message,
      email : email,
      // You had 'email: email,' here. 'from_email' is usually sufficient for EmailJS to get the sender's email.
      // If you need it explicitly as 'email' in your template, keep it, but it's redundant if from_email serves that purpose.
    };

    emailjs.send(
      // Use VITE_ prefix for environment variables with Vite
      import.meta.env.VITE_EMAILJS_SERVICE_ID,
      import.meta.env.VITE_EMAILJS_TEMPLATE_ID,
      templateParams,
      import.meta.env.VITE_EMAILJS_PUBLIC_KEY // This is the Public Key/User ID
    )
    .then(() => {
      setIsSending(false); // <--- Re-added: Set loading to false on success
      toast.success('Message sent successfully!');
      setName('');
      setEmail('');
      setMessage('');
    })
    .catch((error) => {
      setIsSending(false); // <--- Re-added: Set loading to false on error
      console.error("EmailJS Error:", error); // Log the full error for debugging
      toast.error('Failed to send message. Try again later.');
    });
  };

  return (
    <section id="contact" className="min-h-screen bg-gradient-to-br from-gray-900 to-black py-20 pt-24 md:pt-32">
      <Toaster />

      <div className="container mx-auto px-8">
        <div className="text-center mb-16">
          <h2 className="text-5xl font-black text-white mb-6 animate-fadeInUp">
            Get In Touch
          </h2>
          <p className="text-xl text-gray-300 max-w-3xl mx-auto animate-fadeInUp" style={{ animationDelay: '0.2s' }}>
            Have questions or feedback about CineScope? We'd love to hear from you.
          </p>
        </div>
        <div className="grid md:grid-cols-2 gap-16 max-w-6xl mx-auto">
          <div className="space-y-8 animate-fadeInLeft">
            {/* Email Us */}
            <a href={`mailto:${CONTACT_EMAIL}`} className="block">
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full flex items-center justify-center text-white text-xl mr-4">
                    📧
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">Email Us</h3>
                    <p className="text-gray-300">{CONTACT_EMAIL}</p>
                  </div>
                </div>
              </div>
            </a>

            {/* LinkedIn */}
            <a href={LINKEDIN_URL} target="_blank" rel="noopener noreferrer" className="block">
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-green-500 to-emerald-500 rounded-full flex items-center justify-center text-white text-xl mr-4">
                    🔗
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">LinkedIn</h3>
                    <p className="text-gray-300">Connect with Abdul Gouse Syeedy</p>
                  </div>
                </div>
              </div>
            </a>

            {/* GitHub */}
            <a href={GITHUB_URL} target="_blank" rel="noopener noreferrer" className="block">
              <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 hover:bg-white/10 transition-all duration-300">
                <div className="flex items-center mb-4">
                  <div className="w-12 h-12 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-xl mr-4">
                    🐙
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-white">GitHub</h3>
                    <p className="text-gray-300">Explore our code on GitHub</p>
                  </div>
                </div>
              </div>
            </a>
          </div>

          <div className="animate-fadeInUp">
            {/* Contact Form */}
            <form onSubmit={handleSubmit} className="bg-white/5 backdrop-blur-sm rounded-2xl p-8 border border-white/10 space-y-6">
              <div>
                <label htmlFor="name" className="block text-white font-semibold mb-2">Name</label>
                <input
                  type="text"
                  id="name"
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="Your Name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                  disabled={isSending} // <--- Re-added: Disable input while sending
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-white font-semibold mb-2">Email</label>
                <input
                  type="email"
                  id="email"
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition-colors"
                  placeholder="your@email.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  disabled={isSending} // <--- Re-added: Disable input while sending
                />
              </div>
              <div>
                <label htmlFor="message" className="block text-white font-semibold mb-2">Message</label>
                <textarea
                  id="message"
                  rows="4"
                  className="w-full bg-white/10 border border-white/20 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 transition-colors resize-none"
                  placeholder="Tell us about your experience with CineScope..."
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  required
                  disabled={isSending} // <--- Re-added: Disable textarea while sending
                ></textarea>
              </div>
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 rounded-lg font-semibold hover:from-blue-700 hover:to-purple-700 transition-all duration-300 transform hover:scale-105
                           flex items-center justify-center space-x-2" // Added flex for centering
                disabled={isSending} // <--- Re-added: Disable the button while sending
              >
                {isSending ? (
                  <>
                    {/* Tailwind CSS Spinner */}
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Sending...</span>
                  </>
                ) : (
                  'Send Message'
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}

export default ContactSection;