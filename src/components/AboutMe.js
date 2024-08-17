import React, { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';
import { Observer } from 'gsap/Observer';
import { GraduationCap, Briefcase } from 'lucide-react';

import './AboutMeSlider.css';

gsap.registerPlugin(Observer);

const WebsitePreview = ({ url }) => {
  return (
    <iframe
      src={url}
      title="Website Preview"
      width="100%"
      height="100%"
      style={{ border: 'none' }}
    />
  );
};

const AboutMeSlider = () => {
  const [category, setCategory] = useState('coursework');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);
  const containerRef = useRef(null);
  const contentRef = useRef(null);
  const titleRef = useRef(null);
  const numberRef = useRef(null);
  const previewRef = useRef(null);
  const textRef = useRef(null);
  const lastScrollTime = useRef(0);
  const scrollCooldown = 500; // ms

  const categories = {
    coursework: [
      {
        title: "CS61B: Data Structures",
        content: "Covers the implementation and analysis of data structures, including lists, queues, trees, and graphs, along with algorithms for sorting and searching.",
        url: "https://sp24.datastructur.es/"
      },
      {
        title: "CS61C: Machine Structures",
        content: "Introduction to computer architecture, focusing on the relationship between hardware and software. Topics include assembly language, caching, pipelining, and parallel processing.",
        url: "https://cs61c.org/"
      },
      {
        title: "CS161: Computer Security",
        content: "Introduction to computer security, including cryptography, network security, and the analysis of vulnerabilities and defenses.",
        url: "https://fa24.cs161.org/"
      },
      {
        title: "EECS127: Optimization Models in Engineering",
        content: "Introduction to optimization techniques and their applications in engineering, covering linear, nonlinear, and integer programming.",
        url: "https://www2.eecs.berkeley.edu/Courses/EECS127/"
      },
      {
        title: "CS170: Efficient Algorithms and Intractable Problems",
        content: "Studies algorithm design and analysis, including graph algorithms, dynamic programming, and NP-completeness.",
        url: "https://cs170.org/"
      },
      {
        title: "DATA100: Principles and Techniques of Data Science",
        content: "Combines inferential thinking, computational thinking, and real-world relevance to teach the data science process end-to-end.",
        url: "https://ds100.org/"
      },
      {
        title: "CS70: Discrete Mathematics and Probability Theory",
        content: "Focuses on fundamental concepts in discrete mathematics and probability theory, including combinatorics, graph theory, and random variables.",
        url: "https://www.eecs70.org/"
      },
      {
        title: "STAT134: Concepts of Probability",
        content: "Introduction to probability theory, including distributions, expectation, and the law of large numbers.",
        url: "https://www.stat134.org/"
      },
      {
        title: "STAT135: Concepts of Statistics",
        content: "Covers fundamental statistical concepts, including hypothesis testing, regression, and analysis of variance.",
        url: "https://classes.berkeley.edu/content/2024-fall-stat-135-001-lec-001"
      },
      {
        title: "Web Design Decal",
        content: "A student-run course that teaches the fundamentals of web design, covering topics such as HTML, CSS, and JavaScript, with a focus on hands-on projects.",
        url: "https://webdesigndecal.github.io/"
      }
    ],
    experience: [
      {
        title: "Software Developer at AppCensus",
        content: "Developed data scraping and analysis tools to enhance the company's cybersecurity database, focusing on extracting and organizing relevant information from applications.",
        url: "https://www.appcensus.io/"
      },
      {
        title: "Research Assistant at UC Berkeley Radwatch",
        content: "Utilized Raspberry Pi devices to study the inverse square law for radiation, contributing to the development of radiation detection and measurement tools.",
        url: "https://radwatch.berkeley.edu/"
      }
    ]
  };

  const animateContent = (direction, newIndex) => {
    if (isAnimating) return;
    setIsAnimating(true);

    const title = titleRef.current;
    const number = numberRef.current;
    const preview = previewRef.current;
    const text = textRef.current;

    const slideDistance = direction * 100;

    const tl = gsap.timeline({
      onComplete: () => {
        setCurrentIndex(newIndex);
        setIsAnimating(false);
      },
      defaults: { ease: "power2.inOut", duration: 0.6 }
    });

    tl.to([title, number], { x: -slideDistance, opacity: 0 }, 0)
      .to(preview, { x: -slideDistance * 1.5, opacity: 0 }, 0)
      .to(text, { x: slideDistance * 1.5, opacity: 0 }, 0)
      .call(() => setCurrentIndex(newIndex))
      .set([preview, text], { x: slideDistance * 1.5 })
      .set([title, number], { x: slideDistance })
      .to([title, number], { x: 0, opacity: 1 })
      .to([preview, text], { x: 0, opacity: 1, stagger: 0.1 }, "-=0.4");
  };

  const handleNavigation = (direction) => {
    const now = Date.now();
    if (now - lastScrollTime.current < scrollCooldown) {
      return;
    }
    lastScrollTime.current = now;

    if (isAnimating) {
      return;
    }
    const newIndex = (currentIndex + direction + categories[category].length) % categories[category].length;
    animateContent(direction, newIndex);
  };

  useEffect(() => {
    const container = containerRef.current;

    const observer = Observer.create({
      target: container,
      type: "wheel",
      wheelSpeed: -1,
      onWheel: (e) => {
        if (Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
          handleNavigation(e.deltaX > 0 ? -1 : 1);
        }
      },
    });

    return () => {
      observer.kill();
    };
  }, [category, currentIndex, categories, isAnimating]);

  const handleCategoryChange = (newCategory) => {
    if (isAnimating || newCategory === category) return;
    setIsAnimating(true);

    const title = titleRef.current;
    const number = numberRef.current;
    const preview = previewRef.current;
    const text = textRef.current;

    gsap.timeline({
      onComplete: () => {
        setCategory(newCategory);
        setCurrentIndex(0);
        setIsAnimating(false);
      },
      defaults: { ease: "power2.inOut", duration: 0.6 }
    })
    .to([title, number], { x: -100, opacity: 0 }, 0)
    .to([preview, text], { y: 50, opacity: 0, stagger: 0.1 }, 0)
    .call(() => {
      setCategory(newCategory);
      setCurrentIndex(0);
    })
    .set([title, number], { x: 100 })
    .set([preview, text], { y: -50 })
    .to([title, number], { x: 0, opacity: 1 })
    .to([preview, text], { y: 0, opacity: 1, stagger: 0.1 }, "-=0.4");
  };

  return (
    <div ref={containerRef} className="about-me-container" id="about">
      <div className="navigation-overlay left-overlay" onClick={() => handleNavigation(-1)}></div>
      <div className="navigation-overlay right-overlay" onClick={() => handleNavigation(1)}></div>
      <div className="about-me-content-wrapper">
        <div className="about-me-header">
          <h2 ref={titleRef} className="about-me-title">
            {category.toUpperCase()}
            <span ref={numberRef} className="about-me-number">{String(currentIndex + 1).padStart(2, '0')}</span>
          </h2>
          <div className="category-selector">
            <button
              onClick={() => handleCategoryChange('coursework')}
              className={`category-button ${category === 'coursework' ? 'active' : ''}`}
              aria-label="Coursework"
            >
              <GraduationCap size={24} />
              <span>Courses</span>
            </button>
            <button
              onClick={() => handleCategoryChange('experience')}
              className={`category-button ${category === 'experience' ? 'active' : ''}`}
              aria-label="Experience"
            >
              <Briefcase size={24} />
              <span>Experience</span>
            </button>
          </div>
        </div>
        <div ref={contentRef} className="about-me-content">
          <div ref={previewRef} className="about-me-preview">
            <WebsitePreview url={categories[category][currentIndex].url} />
          </div>
          <div ref={textRef} className="about-me-text">
            <h3>{categories[category][currentIndex].title}</h3>
            <p>{categories[category][currentIndex].content}</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AboutMeSlider;