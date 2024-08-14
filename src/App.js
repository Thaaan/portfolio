import React, { useState } from 'react';
import Hero from './components/Hero';
import Header from './components/Header';
import Intro from './components/Intro';
import Projects from './components/Projects';
import Contact from './components/Contact';
import Profile from './components/Profile'
import AboutMe from './components/AboutMe'
import Transition from './components/Transition';
import './App.css';

function App() {
  const [showHero, setShowHero] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showMain, setShowMain] = useState(false);

  const handleHeroAnimationComplete = () => {
    setIsTransitioning(true);
  };

  const handleTransitionComplete = () => {
    setShowHero(false);
    setShowMain(true);
    setIsTransitioning(false);  // Reset the transition state
  };

  return (
    <div className="App">
      {showHero && <Hero onAnimationComplete={handleHeroAnimationComplete} />}
      <Transition
        isTransitioning={isTransitioning}
        onTransitionComplete={handleTransitionComplete}
      />
      {showMain && (
        <>
          <Header />
          <Intro />
          <Projects />
          <Profile />
          <AboutMe />
          <Contact />
        </>
      )}
    </div>
  );
}

export default App;