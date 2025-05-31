import React from 'react'
import ReactDOM from 'react-dom/client'
import HomePage from './pages/HomePage';
import MovieAnalyzer from './components/MovieAnalyzer';
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/analyze" element={<MovieAnalyzer />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>,
)
