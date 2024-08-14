import React, { useEffect, useState } from 'react';
import { gsap, Power3 } from 'gsap';

const Transition = ({ isTransitioning, onTransitionComplete }) => {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    if (isTransitioning) {
      setIsVisible(true);
      const transitionTl = gsap.timeline({
        onComplete: () => {
          onTransitionComplete();
          gsap.to('.transition-rect', {
            width: '0%',
            duration: 0.3,
            ease: Power3.easeInOut,
            stagger: -0.05,
            onComplete: () => {
              setIsVisible(false);
            }
          });
        },
      });

      transitionTl.to('.transition-rect', {
        width: '100%',
        duration: 0.3,
        ease: Power3.easeInOut,
        stagger: 0.05,
      });
    }
  }, [isTransitioning, onTransitionComplete]);

  if (!isVisible) return null;

  return (
    <div className="transition-container">
      <div className="transition-rect"></div>
      <div className="transition-rect"></div>
      <div className="transition-rect"></div>
    </div>
  );
};

export default Transition;