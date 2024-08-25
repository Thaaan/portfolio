import React, { useState, useRef, useCallback } from 'react';
import Slider from 'react-slick';
import Modal from 'react-modal';

import './Projects.css'

const projects = [
  {
    title: 'SmartLinked',
    img: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/smartlinked.png',
    detailedImg: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/smartlinked-collage.png',
    linkTitle: 'Website',
    link: 'https://www.smartlinked.app/',
    description: 'SmartLinked is an AI-powered LinkedIn enhancer that offers personalized profile suggestions to help users optimize their LinkedIn presence.',
    skills: ['React', 'JavaScript', 'CSS', 'HTML', 'PyTorch']
  },
  {
    title: 'KudoTools',
    img: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/kudo.png',
    detailedImg: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/kudo-collage.png',
    linkTitle: 'Github',
    link: 'https://github.com/Kudo-Tools/kudo-tools.github.io',
    description: 'KudoTools is a resource manager designed to assist in purchasing desirable e-commerce items for resale at a higher price. It includes tools like auto captcha solvers, recaptcha bypasses, and more.',
    skills: ['Java', 'Python', 'PHP', 'CSS', 'HTML', 'JavaScript', 'Selenium']
  },
  {
    title: 'Portfolio',
    img: 'https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/projects/portfolio.png',
    detailedImg: 'https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/projects/portfolio-collage.png',
    linkTitle: 'Github',
    link: 'https://github.com/Thaaan/portfolio',
    description: 'This portfolio website highlights my software development skills, projects, and professional journey. It features a modern, responsive design with interactive elements, including a real-time MNIST digit classifier demo, showcasing my expertise in front-end development, back-end integration, and machine learning.',
    skills: ['React', 'JavaScript', 'HTML', 'CSS', 'Python', 'Flask', 'RESTful API', 'Machine Learning', 'TensorFlow', 'Redis', 'GSAP']
  },
  {
    title: 'ThaanAquatics',
    img: 'https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/projects/thaanaquatics.png',
    detailedImg: 'https://ethirwin-portfolio-assets.s3.us-east-2.amazonaws.com/img/projects/thaanaquatics-collage.png',
    linkTitle: 'Github',
    link: 'https://github.com/Thaaan/ThaanAquatics',
    description: 'ThaanAquatics is an e-commerce platform for selling aquarium supplies and live fish. It features an intuitive user interface, efficient product categorization, and a secure payment system for customers. The website allows users to browse various aquarium supplies and purchase live fish, with integration for inventory management and shipping logistics.',
    skills: ['JavaScript', 'CSS', 'HTML', 'EJS', 'Express', 'MySQL', 'SQL']
  }
];

const ProjectCarousel = () => {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const sliderRef = useRef(null);

  const preloadImage = (src) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = src;
      img.onload = resolve;
      img.onerror = reject;
    });
  };

  const openModal = useCallback(async (project) => {
    setSelectedProject(project);
    setIsImageLoaded(false);
    try {
      await preloadImage(project.detailedImg);
      setIsImageLoaded(true);
      setModalIsOpen(true);
    } catch (error) {
      console.error('Failed to load image:', error);
      // Optionally, you can still open the modal or show an error message
      setModalIsOpen(true);
    }
  }, []);

  const closeModal = () => {
    setModalIsOpen(false);
    setSelectedProject(null);
    setIsImageLoaded(false);
  };

  const handleSlideClick = (index, project) => {
    if (index === currentIndex) {
      openModal(project);
    } else {
      const totalSlides = projects.length;
      const half = Math.floor(totalSlides / 2);
      const diff = index - currentIndex;

      if (Math.abs(diff) <= half) {
        sliderRef.current.slickGoTo(index);
      } else {
        if (diff > 0) {
          sliderRef.current.slickGoTo(currentIndex - (totalSlides - diff));
        } else {
          sliderRef.current.slickGoTo(currentIndex + (totalSlides + diff));
        }
      }
    }
  };

  const settings = {
    dots: true,
    infinite: true,
    speed: 500,
    slidesToShow: 3,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 4000,
    pauseOnHover: true,
    centerMode: true,
    centerPadding: '0',
    beforeChange: (oldIndex, newIndex) => {
      setCurrentIndex(newIndex);
    },
    responsive: [
      {
        breakpoint: 1200,
        settings: {
          slidesToShow: 2,
          slidesToScroll: 1,
          centerMode: false,
        }
      },
      {
        breakpoint: 768,
        settings: {
          slidesToShow: 1,
          slidesToScroll: 1,
          centerMode: false,
        }
      }
    ]
  };

  return (
    <section id="projects">
      <Slider {...settings} className="projects-carousel" ref={sliderRef}>
        {projects.map((project, index) => (
          <div key={index} className="card" onClick={() => handleSlideClick(index, project)}>
            <img src={project.img} alt={project.title} />
          </div>
        ))}
      </Slider>
      {selectedProject && (
        <Modal
          isOpen={modalIsOpen && isImageLoaded}
          onRequestClose={closeModal}
          contentLabel="Project Details"
          className="modal"
          overlayClassName="overlay"
        >
          <div className="modal-content">
            <button onClick={closeModal} className="close-button">×</button>
            <div className="modal-left">
              <img src={selectedProject.detailedImg} alt={selectedProject.title} className="detailed-img" />
            </div>
            <div className="modal-right">
              <div className="modal-header">
                <h2>{selectedProject.title}</h2>
              </div>
              <a href={selectedProject.link} target="_blank" rel="noreferrer" className="link">{selectedProject.linkTitle}</a>
              <p>{selectedProject.description}</p>
              <div className="skills">
                {selectedProject.skills.map((skill, index) => (
                  <span key={index} className="skill-box">{skill}</span>
                ))}
              </div>
            </div>
          </div>
        </Modal>
      )}
    </section>
  );
};

export default ProjectCarousel;