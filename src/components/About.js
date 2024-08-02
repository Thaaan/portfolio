import React from 'react';
import MNISTClassifier from './MNISTClassifier'

//import images
import githubIcon from './../img/icons/github-logo.png'
import linkedinIcon from './../img/icons/linkedin.png'
import twitterIcon from './../img/icons/twitter.png'
import instaIcon from './../img/icons/instagram.png'

const About = () => {
  const socialLinks = [
    { name: 'GitHub', url: 'https://github.com/Thaaan', icon: githubIcon },
    { name: 'LinkedIn', url: 'https://www.linkedin.com/in/ethirwin/', icon: linkedinIcon },
    { name: 'Twitter', url: 'https://x.com/arkto_', icon: twitterIcon },
    { name: 'Instagram', url: 'https://www.instagram.com/ethan.irw/', icon: instaIcon },
  ];

  return (
    <section id="about">
      <div className="about-content">
        <h1 className="welcome-text">Welcome</h1>
        <h2 className="name-text">I'm Ethan</h2>
        <p className="about-description">
          [Your brief introduction or tagline goes here. Make it catchy and informative!]
        </p>
        <div className="social-links">
          {socialLinks.map((link, index) => (
            <a key={index} href={link.url} target="_blank" rel="noopener noreferrer" className="social-icon">
              <img src={link.icon} alt={link.name} />
            </a>
          ))}
        </div>
      </div>
      <div className="about-demo">
        <MNISTClassifier />
      </div>
    </section>
  );
};

export default About;