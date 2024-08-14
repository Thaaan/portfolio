import React from 'react';
import { GraduationCap, Code, Wrench } from 'lucide-react';
import './ProfileComponent.css';

const ProfileComponent = () => {
  const languages = ['Java', 'C', 'Python', 'React', 'HTML', 'CSS', 'JS', 'SQL', 'PHP'];
  const tools = ['Git', 'Selenium', 'TensorFlow', 'PyTorch'];

  return (
    <div className="profile-container" id="profile">
      <div className="profile-card">
        <div className="school-info">
          <GraduationCap size={32} />
          <h2>UC Berkeley</h2>
          <h3>B.A. Computer Science and Statistics</h3>
          <p>Rigorous coursework in algorithms, data structures, and statistical analysis. Developed strong problem-solving skills and a deep understanding of computational theory and practical applications.</p>
        </div>
        <div className="skills-info">
          <div className="languages-section">
            <h3><Code size={24} /> Programming Languages</h3>
            <p>The various languages I have practiced coding and feel confident using.</p>
            <div className="language-boxes">
              {languages.map((lang, index) => (
                <span key={index} className="language-box">{lang}</span>
              ))}
            </div>
          </div>
          <div className="tools-section">
            <h3><Wrench size={24} /> Tools</h3>
            <p>The different tools I use for efficient and effective programming.</p>
            <div className="tool-boxes">
              {tools.map((tool, index) => (
                <span key={index} className="tool-box">{tool}</span>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileComponent;
