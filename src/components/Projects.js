import React, { useState, useRef } from 'react';
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
    skills: ['React', 'Node.js', 'AI', 'Machine Learning']
  },
  {
    title: 'KudoTools',
    img: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/kudo.png',
    detailedImg: 'https://d37cdst5t0g8pp.cloudfront.net/img/projects/kudo-collage.png',
    linkTitle: 'Github',
    link: 'https://github.com/Kudo-Tools/kudo-tools.github.io',
    description: 'KudoTools is a resource manager designed to assist in purchasing desirable e-commerce items for resale at a higher price. It includes tools like auto captcha solvers, recaptcha bypasses, and more.',
    skills: ['Python', 'Web Scraping', 'Automation', 'Security']
  },
  {
    title: 'Project 3',
    img: 'https://via.placeholder.com/800x400',
    detailedImg: 'https://via.placeholder.com/400x800',
    linkTitle: '',
    link: '#',
    description: 'Detailed information about Project 3',
    skills: ['Skill1', 'Skill2', 'Skill3']
  },
  {
    title: 'Project 4',
    img: 'https://via.placeholder.com/800x400',
    detailedImg: 'https://via.placeholder.com/400x800',
    linkTitle: '',
    link: '#',
    description: 'Detailed information about Project 4',
    skills: ['Skill1', 'Skill2', 'Skill3']
  },
];

const ProjectCarousel = () => {
  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState(null);
  const [currentIndex, setCurrentIndex] = useState(0);
  const sliderRef = useRef(null);

  const openModal = (project) => {
    setSelectedProject(project);
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
    setSelectedProject(null);
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
          isOpen={modalIsOpen}
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