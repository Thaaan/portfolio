import React from 'react';
import MNISTClassifier from './MNISTClassifier'

import './Intro.css'

const Intro = () => {
  const socialLinks = [
    { name: 'GitHub', url: 'https://github.com/Thaaan', icon: 'https://d37cdst5t0g8pp.cloudfront.net/img/icons/github-logo.png' },
    { name: 'LinkedIn', url: 'https://www.linkedin.com/in/ethirwin/', icon: 'https://d37cdst5t0g8pp.cloudfront.net/img/icons/linkedin.png' },
    { name: 'Twitter', url: 'https://x.com/arkto_', icon: 'https://d37cdst5t0g8pp.cloudfront.net/img/icons/twitter.png' },
    { name: 'Instagram', url: 'https://www.instagram.com/ethan.irw/', icon: 'https://d37cdst5t0g8pp.cloudfront.net/img/icons/instagram.png' },
  ];

  return (
    <section id="intro">
      <div className="intro-content">
        <h1 className="welcome-text">Welcome</h1>
        <h2 className="name-text">I'm Ethan</h2>
        <p className="intro-description">
          A passionate software engineer and avid problem solver. Check out the simple digit classifier demo on the right!
        </p>
        <div className="social-links">
          {socialLinks.map((link, index) => (
            <a key={index} href={link.url} target="_blank" rel="noopener noreferrer" className="social-icon">
              <img src={link.icon} alt={link.name} />
            </a>
          ))}
          <a className="resume-link" href="https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/Resume.pdf" target="_blank" rel="noopener noreferrer">Resume</a>
        </div>
      </div>
      <div className="intro-demo">
        <MNISTClassifier />
      </div>
    </section>
  );
};

export default Intro;