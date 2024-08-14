import React, { useReducer, useEffect, useCallback, useRef, useState } from 'react';
import io from 'socket.io-client';

const API_URL = 'https://portfolio-ethan-06ccc7665c32.herokuapp.com';

const initialState = {
  prediction: null,
  isTraining: true,
  trainingLogs: [],
  error: null,
  isLoading: false,
};

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_LOG':
      return { ...state, trainingLogs: [...state.trainingLogs, action.payload] };
    case 'SET_TRAINING':
      return { ...state, isTraining: action.payload };
    case 'SET_PREDICTION':
      return { ...state, prediction: action.payload };
    case 'RESET_LOGS':
      return { ...state, trainingLogs: [] };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };
    default:
      return state;
  }
}

const MNISTClassifier = () => {
  const [state, dispatch] = useReducer(reducer, initialState);
  const [commandHistory, setCommandHistory] = useState([]);
  const [inputVisible, setInputVisible] = useState(true);
  const [showCanvas, setShowCanvas] = useState(false);
  const socketRef = useRef(null);
  const terminalRef = useRef(null);
  const inputRef = useRef(null);
  const stateRef = useRef(state);

  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  useEffect(() => {
    socketRef.current = io(API_URL);

    socketRef.current.on('training_log', (data) => {
      dispatch({ type: 'ADD_LOG', payload: data.data });
      if (data.data.includes('Finished Training')) {
        dispatch({ type: 'SET_TRAINING', payload: false });
      }
    });

    socketRef.current.on('connect', () => {
      dispatch({ type: 'SET_TRAINING', payload: true });
      dispatch({ type: 'RESET_LOGS' });
    });

    socketRef.current.on('connect_error', (error) => {
      dispatch({ type: 'SET_ERROR', payload: 'Failed to connect to server' });
    });

    return () => {
      socketRef.current.disconnect();
    };
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [state.trainingLogs, commandHistory]);

  useEffect(() => {
    if (!state.isTraining) {
      setTimeout(() => {
        dispatch({ type: 'ADD_LOG', payload: 'Switching to canvas view...' });
        setTimeout(() => {
          setShowCanvas(true);
        }, 2000);
      }, 2000);
    }
  }, [state.isTraining]);

  const startTraining = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    try {
      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error('Failed to start training');
      }
      const data = await response.json();
      dispatch({ type: 'ADD_LOG', payload: data.message });
      if (data.message === 'Model already trained') {
        dispatch({ type: 'SET_TRAINING', payload: false });
      }
    } catch (error) {
      console.error('Error:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && inputRef.current.value.toLowerCase() === 'start training model') {
      startTraining();
      setCommandHistory([...commandHistory, inputRef.current.value]);
      setInputVisible(false);
      inputRef.current.value = '';
    }
  };

  useEffect(() => {
    if (inputRef.current && inputVisible) {
      inputRef.current.focus();
    }
  }, [state.isTraining, inputVisible]);

  const Terminal = () => (
    <div className="terminal-window">
      <div className="terminal-header">
        <div className="terminal-buttons">
          <div className="terminal-button close"></div>
          <div className="terminal-button minimize"></div>
          <div className="terminal-button maximize"></div>
        </div>
        <div className="terminal-title">MNIST Digit Classifier Training</div>
      </div>
      <div className="terminal-container" ref={terminalRef}>
        {commandHistory.map((cmd, index) => (
          <div key={index} className="terminal-command">
            <span className="terminal-prompt">$</span>
            <p className="terminal-input">{cmd}</p>
          </div>
        ))}
        {state.trainingLogs.map((log, index) => (
          <div key={index} className="terminal-log">{log}</div>
        ))}
        {inputVisible && (
          <div className="terminal-command">
            <span className="terminal-prompt">$</span>
            <input
              ref={inputRef}
              type="text"
              className="terminal-input"
              onKeyPress={handleKeyPress}
              aria-label="Command input"
              placeholder="start training model"
            />
          </div>
        )}
      </div>
    </div>
  );

  const DrawingCanvas = () => {
    const [canvasKey, setCanvasKey] = useState(0);
    const canvasRef = useRef(null);
    const ctxRef = useRef(null);
    const isDrawingRef = useRef(false);
    const canvasContainerRef = useRef(null);
    const lastPointRef = useRef(null);

    useEffect(() => {
      if (canvasRef.current) {
        initializeCanvas();
        setupEventListeners();
      }
    }, [canvasKey]);

    const initializeCanvas = () => {
      const canvas = canvasRef.current;
      const container = canvasContainerRef.current;
      if (!canvas || !container) return;

      const ctx = canvas.getContext('2d');
      ctxRef.current = ctx;

      canvas.width = 280;
      canvas.height = 280;

      canvas.style.width = '100%';
      canvas.style.height = '100%';

      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 10;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      canvas.style.cursor = 'crosshair';
    };

    const setupEventListeners = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const getMousePos = (e) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
          x: (e.clientX - rect.left) * scaleX,
          y: (e.clientY - rect.top) * scaleY
        };
      };

      const startDrawing = (e) => {
        isDrawingRef.current = true;
        lastPointRef.current = getMousePos(e);
      };

      const draw = (e) => {
        if (!isDrawingRef.current) return;
        const newPoint = getMousePos(e);
        const ctx = ctxRef.current;

        ctx.beginPath();
        ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);

        // Line interpolation
        const points = interpolatePoints(lastPointRef.current, newPoint);
        points.forEach(point => {
          ctx.lineTo(point.x, point.y);
        });

        ctx.stroke();
        lastPointRef.current = newPoint;
      };

      const stopDrawing = () => {
        isDrawingRef.current = false;
      };

      canvas.addEventListener('mousedown', startDrawing);
      canvas.addEventListener('mousemove', draw);
      canvas.addEventListener('mouseup', stopDrawing);
      canvas.addEventListener('mouseout', stopDrawing);

      canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        startDrawing(e.touches[0]);
      });
      canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        draw(e.touches[0]);
      });
      canvas.addEventListener('touchend', stopDrawing);
    };

    const interpolatePoints = (p1, p2) => {
      const dx = p2.x - p1.x;
      const dy = p2.y - p1.y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const steps = Math.max(Math.floor(distance / 2), 1);
      const points = [];

      for (let i = 0; i < steps; i++) {
        points.push({
          x: p1.x + (dx * i) / steps,
          y: p1.y + (dy * i) / steps
        });
      }

      return points;
    };

    const clearCanvas = useCallback(() => {
      if (!canvasRef.current || !ctxRef.current) return;
      ctxRef.current.fillStyle = 'black';
      ctxRef.current.fillRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }, []);

    const resetCanvas = useCallback(() => {
      clearCanvas();
      setCanvasKey(prevKey => prevKey + 1);
    }, [clearCanvas]);

    const classifyDigit = useCallback(async () => {
      if (!canvasRef.current) return;
      dispatch({ type: 'SET_LOADING', payload: true });

      const canvas = canvasRef.current;
      const imageData = canvas.toDataURL('image/png');

      try {
        const response = await fetch(`${API_URL}/predict`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ image: imageData }),
        });

        if (!response.ok) {
          throw new Error('Failed to classify digit');
        }

        const data = await response.json();
        dispatch({ type: 'SET_PREDICTION', payload: data.prediction });
      } catch (error) {
        console.error('Error:', error);
        dispatch({ type: 'SET_ERROR', payload: error.message });
      } finally {
        dispatch({ type: 'SET_LOADING', payload: false });
      }
    }, []);

    return (
      <div className="canvas-section">
        <div className="drawing-area" ref={canvasContainerRef}>
          <canvas
            key={canvasKey}
            ref={canvasRef}
            className="drawing-canvas"
            aria-label="Drawing canvas for digit classification"
          />
        </div>
        <div className="canvas-sidebar">
          <div>
            <h2 className="canvas-title">MNIST Digit Classifier</h2>
            <div className="canvas-controls">
              <button
                onClick={resetCanvas}
                className="control-button"
                aria-label="Clear canvas"
              >
                Clear
              </button>
              <button
                onClick={classifyDigit}
                className="control-button"
                disabled={state.isLoading}
                aria-label="Classify digit"
              >
                {state.isLoading ? 'Classifying...' : 'Classify'}
              </button>
            </div>
          </div>
          {state.prediction !== null && (
            <div className="prediction-result" aria-live="polite">
              Predicted digit: <span>{state.prediction}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="mnist-classifier">
      {state.error && (
        <div className="error-message" role="alert">
          <strong>Error:</strong> {state.error}
        </div>
      )}
      {showCanvas ? <DrawingCanvas /> : <Terminal />}
    </div>
  );
};

export default MNISTClassifier;