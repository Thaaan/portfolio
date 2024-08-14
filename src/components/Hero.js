import React, { useEffect } from 'react';
import { gsap, Power3 } from 'gsap';

const Hero = ({ onAnimationComplete }) => {
  useEffect(() => {
    const tl = gsap.timeline({
      onComplete: onAnimationComplete,
    });

    tl.fromTo(
      '.word-1',
      { x: -200, opacity: 0 },
      { x: 0, opacity: 1, duration: 0.8, ease: Power3.easeOut }
    )
      .fromTo(
        '.word-2',
        { x: 200, opacity: 0 },
        { x: 0, opacity: 1, duration: 0.8, ease: Power3.easeOut },
        '-=0.6'
      )
      .fromTo(
        '.word-3',
        { y: -200, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: Power3.easeOut },
        '-=0.6'
      )
      .fromTo(
        '.word-4',
        { y: 200, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, ease: Power3.easeOut },
        '-=0.6'
      );
  }, [onAnimationComplete]);

  return (
    <div className="hero">
      <div className="hero-content">
        <h1 className="word word-1">Student</h1>
        <h1 className="word word-2">Developer</h1>
        <h1 className="word word-3">Explorer</h1>
        <h1 className="word word-4">Research</h1>
      </div>
    </div>
  );
};

export default Hero;