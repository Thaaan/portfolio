import React, { useState, useEffect, useRef, useCallback } from 'react';
import { gsap } from 'gsap';
import { Observer } from 'gsap/Observer';
import { GraduationCap, Briefcase, ChevronLeft, ChevronRight } from 'lucide-react';

import './AboutMeSlider.css';

gsap.registerPlugin(Observer);

const WebsitePreview = React.memo(({ url, previewImageUrl, isVisible }) => {
  const handleClick = useCallback(() => {
    window.open(url, '_blank', 'noopener,noreferrer');
  }, [url]);

  return (
    <div
      className={`website-preview ${isVisible ? 'visible' : ''}`}
      onClick={handleClick}
    >
      <img
        src={previewImageUrl}
        alt="Website Preview"
        className="website-preview-image"
      />
    </div>
  );
});

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
  const touchStartX = useRef(null);

  const categories = {
    coursework: [
      {
        title: "CS61B: Data Structures",
        content: "Covers the implementation and analysis of data structures, including lists, queues, trees, and graphs, along with algorithms for sorting and searching.",
        url: "https://sp24.datastructur.es/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/61b.png"
      },
      {
        title: "CS61C: Machine Structures",
        content: "Introduction to computer architecture, focusing on the relationship between hardware and software. Topics include assembly language, caching, pipelining, and parallel processing.",
        url: "https://cs61c.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/61c.png"
      },
      {
        title: "CS161: Computer Security",
        content: "Introduction to computer security, including cryptography, network security, and the analysis of vulnerabilities and defenses.",
        url: "https://fa24.cs161.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/cs161.png"
      },
      {
        title: "EECS127: Optimization Models in Engineering",
        content: "Introduction to optimization techniques and their applications in engineering, covering linear, nonlinear, and integer programming.",
        url: "https://www2.eecs.berkeley.edu/Courses/EECS127/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/eecs127.png"
      },
      {
        title: "CS170: Efficient Algorithms and Intractable Problems",
        content: "Studies algorithm design and analysis, including graph algorithms, dynamic programming, and NP-completeness.",
        url: "https://cs170.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/cs170.png"
      },
      {
        title: "DATA100: Principles and Techniques of Data Science",
        content: "Combines inferential thinking, computational thinking, and real-world relevance to teach the data science process end-to-end.",
        url: "https://ds100.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/data100.png"
      },
      {
        title: "CS70: Discrete Mathematics and Probability Theory",
        content: "Focuses on fundamental concepts in discrete mathematics and probability theory, including combinatorics, graph theory, and random variables.",
        url: "https://www.eecs70.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/cs70.png"
      },
      {
        title: "STAT134: Concepts of Probability",
        content: "Introduction to probability theory, including distributions, expectation, and the law of large numbers.",
        url: "https://www.stat134.org/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/stat134.png"
      },
      {
        title: "STAT135: Concepts of Statistics",
        content: "Covers fundamental statistical concepts, including hypothesis testing, regression, and analysis of variance.",
        url: "https://classes.berkeley.edu/content/2024-fall-stat-135-001-lec-001",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/stat135.png"
      },
      {
        title: "Web Design Decal",
        content: "A student-run course that teaches the fundamentals of web design, covering topics such as HTML, CSS, and JavaScript, with a focus on hands-on projects.",
        url: "https://webdesigndecal.github.io/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/coursework/wdd.png"
      }
    ],
    experience: [
      {
        title: "Software Developer at AppCensus",
        content: "Developed data scraping and analysis tools to enhance the company's cybersecurity database, focusing on extracting and organizing relevant information from applications.",
        url: "https://www.appcensus.io/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/experience/appcensus.png"
      },
      {
        title: "Research Assistant at UC Berkeley Radwatch",
        content: "Utilized Raspberry Pi devices to study the inverse square law for radiation, contributing to the development of radiation detection and measurement tools.",
        url: "https://radwatch.berkeley.edu/",
        previewImg: "https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/experience/radwatch.png"
      }
    ]
  };

  const getAdjacentIndices = (index, length) => {
    const prevIndex = (index - 1 + length) % length;
    const nextIndex = (index + 1) % length;
    return [prevIndex, index, nextIndex];
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

    const handleTouchStart = (e) => {
      touchStartX.current = e.touches[0].clientX;
    };

    const handleTouchEnd = (e) => {
      if (touchStartX.current === null) return;

      const touchEndX = e.changedTouches[0].clientX;
      const diff = touchStartX.current - touchEndX;

      if (Math.abs(diff) > 50) { // Adjust this threshold as needed
        handleNavigation(diff > 0 ? 1 : -1);
      }

      touchStartX.current = null;
    };

    container.addEventListener('touchstart', handleTouchStart);
    container.addEventListener('touchend', handleTouchEnd);

    return () => {
      observer.kill();
      container.removeEventListener('touchstart', handleTouchStart);
      container.removeEventListener('touchend', handleTouchEnd);
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
            {getAdjacentIndices(currentIndex, categories[category].length).map((index) => (
              <WebsitePreview
                key={index}
                url={categories[category][index].url}
                previewImageUrl={categories[category][index].previewImg}
                isVisible={index === currentIndex}
              />
            ))}
          </div>
          <div ref={textRef} className="about-me-text">
            <h3>{categories[category][currentIndex].title}</h3>
            <p>{categories[category][currentIndex].content}</p>
          </div>
        </div>
        <div className="navigation-buttons">
          <button className="nav-button prev" onClick={() => handleNavigation(-1)} aria-label="Previous">
            <ChevronLeft size={24} />
          </button>
          <button className="nav-button next" onClick={() => handleNavigation(1)} aria-label="Next">
            <ChevronRight size={24} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default AboutMeSlider;