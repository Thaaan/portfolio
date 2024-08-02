import React, { useState, useRef } from 'react';
import Slider from 'react-slick';
import Modal from 'react-modal';

//import images
import smartlinkedPreview from './../img/smartlinked.png'
import smartlinkedDashboard from './../img/smartlinked-dashboard.png'

const projects = [
  {
    title: 'Project 1',
    img: smartlinkedDashboard,
    detailedImg: smartlinkedPreview,
    link: '#',
    description: 'Detailed information about Project 1',
    github: 'https://github.com/project1',
  },
  {
    title: 'Project 2',
    img: 'https://via.placeholder.com/800x400',
    detailedImg: 'https://via.placeholder.com/800x400',
    link: '#',
    description: 'Detailed information about Project 2',
    github: 'https://github.com/project2',
  },
  {
    title: 'Project 3',
    img: 'https://via.placeholder.com/800x400',
    detailedImg: 'https://via.placeholder.com/800x400',
    link: '#',
    description: 'Detailed information about Project 3',
    github: 'https://github.com/project3',
  },
  {
    title: 'Project 4',
    img: 'https://via.placeholder.com/800x400',
    detailedImg: 'https://via.placeholder.com/800x400',
    link: '#',
    description: 'Detailed information about Project 4',
    github: 'https://github.com/project4',
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
  };

  return (
    <section id="projects">
      <Slider {...settings} className="projects-carousel" ref={sliderRef}>
        {projects.map((project, index) => (
          <div key={index} className="card" onClick={() => handleSlideClick(index, project)}>
            <a href={project.link}>
              <img src={project.img} alt={project.title} />
            </a>
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
              <a href={selectedProject.github} className="github-link">Github</a>
              <p>{selectedProject.description}</p>
            </div>
          </div>
        </Modal>
      )}
    </section>
  );
};

export default ProjectCarousel;
