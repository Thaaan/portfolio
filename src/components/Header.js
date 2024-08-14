import React, { useEffect, useRef, useState } from 'react';
import { gsap } from 'gsap';
import './Header.css'

const Navbar = () => {
  const navRef = useRef(null);
  const tlRef = useRef(null);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    tlRef.current = gsap.timeline({ defaults: { duration: 0.5, ease: 'power2.inOut' } });
    tlRef.current.pause();

    tlRef.current
      .to(navRef.current, { right: 0, duration: 0.3 })
      .to(navRef.current, { height: '100vh' }, '-=0.1')
      .to('.nav-items li', { opacity: 1, y: 0, stagger: 0.05 }, '-=0.3')

    return () => {
      tlRef.current.kill();
    };
  }, []);

  const toggleMenu = () => {
    setIsOpen(!isOpen);
    if (tlRef.current.reversed() || tlRef.current.paused()) {
      tlRef.current.play();
    } else {
      tlRef.current.reverse();
    }
  };

  const handleNavClick = (sectionId) => {
    let selectedSection = sectionId;
    toggleMenu();
    tlRef.current.reverse().eventCallback('onReverseComplete', () => {
      if (selectedSection) {
        const section = document.getElementById(selectedSection);
        if (section) {
          section.scrollIntoView({ behavior: 'smooth' });
        }
        selectedSection = null;
      }
    });
  };

  return (
    <>
      <button
        className={`menu-toggle ${isOpen ? 'open' : ''}`}
        onClick={toggleMenu}
        aria-label={isOpen ? "Close menu" : "Open menu"}
      >
        <div className="menu-icon">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </button>
      <nav ref={navRef} className="nav-menu">
        <ul className="nav-items">
          <li>
            <a onClick={() => handleNavClick('intro')}>
              <span className="nav-item-number">01</span>
              <span className="nav-item-text">Home</span>
              <span className="nav-item-dot"></span>
            </a>
          </li>
          <li>
            <a onClick={() => handleNavClick('projects')}>
              <span className="nav-item-number">02</span>
              <span className="nav-item-text">Projects</span>
              <span className="nav-item-dot"></span>
            </a>
          </li>
          <li>
            <a onClick={() => handleNavClick('profile')}>
              <span className="nav-item-number">03</span>
              <span className="nav-item-text">About</span>
              <span className="nav-item-dot"></span>
            </a>
          </li>
          <li>
            <a onClick={() => handleNavClick('contact')}>
              <span className="nav-item-number">04</span>
              <span className="nav-item-text">Contact</span>
              <span className="nav-item-dot"></span>
            </a>
          </li>
        </ul>
      </nav>
    </>
  );
};

export default Navbar;