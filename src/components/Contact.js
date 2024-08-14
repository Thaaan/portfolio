import React, { useState, useRef, useEffect } from 'react';
import { Send, Check, User, Bot } from 'lucide-react';
import './Contact.css';

const API_URL = process.env.API_URL;

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  });

  const [errors, setErrors] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');

  const chatboxRef = useRef(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value,
    });
    // Clear error for this field when user starts typing
    if (errors[name]) {
      setErrors({
        ...errors,
        [name]: null,
      });
    }
  };

  const validate = () => {
    let tempErrors = {};
    if (!formData.name.trim()) tempErrors.name = 'Name is required';
    if (!formData.email.trim()) {
      tempErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      tempErrors.email = 'Email address is invalid';
    }
    if (!formData.message.trim()) tempErrors.message = 'Message is required';
    setErrors(tempErrors);
    return Object.keys(tempErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (validate()) {
      setIsSubmitting(true);
      setSubmitError(null);
      try {
        const response = await fetch(`${API_URL}/email`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(formData),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Failed to send email');
        }

        setIsSubmitted(true);
        setFormData({ name: '', email: '', message: '' });

        // Reset form after 3 seconds
        setTimeout(() => {
          setIsSubmitted(false);
        }, 3000);
      } catch (error) {
        console.error('Error sending email:', error);
        setSubmitError(error.message || 'There was an error sending your message. Please try again.');
      } finally {
        setIsSubmitting(false);
      }
    }
  };

  const handleChatSubmit = (e) => {
    e.preventDefault();
    if (chatInput.trim()) {
      setChatMessages(prevMessages => [...prevMessages, { sender: 'user', text: chatInput }]);
      setChatInput('');

      // TODO: replace placeholder for bot response
      setTimeout(() => {
        setChatMessages(prevMessages => [...prevMessages, { sender: 'bot', text: 'This is a placeholder response.' }]);
      }, 1000);
    }
  };

  const scrollToBottom = () => {
    if (chatboxRef.current) {
      chatboxRef.current.scrollTop = chatboxRef.current.scrollHeight;
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [chatMessages]);

  return (
    <section id="contact" className="contact-section">
      <div className="contact-container">
        <div className="contact-header">
          <h2 className="contact-title">Contact Me</h2>
          <p>If you have any questions or just want to get in touch, feel free to send a message or chat with me below.</p>
        </div>
        <div className="contact-content">
          <div className="contact-form-container">
            <form onSubmit={handleSubmit} className="contact-form">
              <div className="form-group">
                <label htmlFor="name">Name:</label>
                <input
                  type="text"
                  id="name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  className={errors.name ? 'error-input' : ''}
                />
                {errors.name && <span className="error">{errors.name}</span>}
              </div>
              <div className="form-group">
                <label htmlFor="email">Email:</label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className={errors.email ? 'error-input' : ''}
                />
                {errors.email && <span className="error">{errors.email}</span>}
              </div>
              <div className="form-group">
                <label htmlFor="message">Message:</label>
                <textarea
                  id="message"
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  className={errors.message ? 'error-input' : ''}
                ></textarea>
                {errors.message && (
                  <span className="error">{errors.message}</span>
                )}
              </div>
              {submitError && (
                <div className="error-message">{submitError}</div>
              )}
              <div className="button-container">
                <button
                  type="submit"
                  className={`submit-button ${isSubmitting ? 'submitting' : ''} ${isSubmitted ? 'submitted' : ''}`}
                  disabled={isSubmitting || isSubmitted}
                >
                  <span className="button-text">
                    {isSubmitting ? 'Sending...' : isSubmitted ? 'Sent!' : 'Send Message'}
                  </span>
                  <span className="button-icon">
                    {isSubmitted && <Check size={18} />}
                  </span>
                </button>
              </div>
            </form>
          </div>
          <div className="chatbox-container">
            <div className="chatbox">
              <div className="chatbox-header">
                <Bot size={24} />
                <span>Chat Assistant</span>
              </div>
              <div className="chatbox-messages" ref={chatboxRef}>
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={`chat-message ${msg.sender === 'bot' ? 'bot' : 'user'}`}
                  >
                    <div className="message-icon">
                      {msg.sender === 'bot' ? <Bot size={16} /> : <User size={16} />}
                    </div>
                    <div className="message-content">{msg.text}</div>
                  </div>
                ))}
              </div>
              <form onSubmit={handleChatSubmit} className="chat-input-form">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Type a message..."
                />
                <button type="submit" className="chat-submit-button">
                  <Send size={18} />
                </button>
              </form>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Contact;