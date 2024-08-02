import React from 'react';

const Header = () => {
  const handleScroll = (sectionId) => {
    const section = document.getElementById(sectionId);
    if (section) {
      section.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <header>
      <nav>
        <a href="#about" onClick={(e) => { e.preventDefault(); handleScroll('about'); }}>About</a>
        <a href="#projects" onClick={(e) => { e.preventDefault(); handleScroll('projects'); }}>Projects</a>
        <a href="#contact" onClick={(e) => { e.preventDefault(); handleScroll('contact'); }}>Contact</a>
      </nav>
    </header>
  );
};

export default Header;
