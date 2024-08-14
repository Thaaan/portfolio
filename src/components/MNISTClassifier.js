import React, { useReducer, useEffect, useCallback, useRef, useState } from 'react';

const API_URL = process.env.REACT_APP_API_URL;

const initialState = {
  prediction: null,
  trainingStatus: 'idle', // 'idle', 'training', 'trained'
  trainingLogs: [],
  error: null,
  isLoading: false,
};

function reducer(state, action) {
  switch (action.type) {
    case 'ADD_LOG':
      return { ...state, trainingLogs: [...state.trainingLogs, action.payload] };
    case 'SET_TRAINING_STATUS':
      return { ...state, trainingStatus: action.payload };
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
  const [showCanvas, setShowCanvas] = useState(false);
  const [inputVisible, setInputVisible] = useState(true);
  const terminalRef = useRef(null);
  const inputRef = useRef(null);
  const eventSourceRef = useRef(null);

  //send heartbeat to confirm user is still active
  useEffect(() => {
    const heartbeatInterval = setInterval(() => {
      fetch(`${API_URL}/heartbeat`, {
        method: 'POST',
        credentials: 'include'
      }).catch((error) => {
        console.error('Heartbeat error:', error);
      });
    }, 60000);

    return () => {
      clearInterval(heartbeatInterval);
    };
  }, []);

  useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight;
    }
  }, [state.trainingLogs, commandHistory]);

  useEffect(() => {
    if (state.trainingStatus === 'trained' && !showCanvas) {
      setTimeout(() => {
        dispatch({ type: 'ADD_LOG', payload: 'Switching to canvas view...' });
        setTimeout(() => {
          setShowCanvas(true);
        }, 2000);
      }, 2000);
    }
  }, [state.trainingStatus, showCanvas]);

  // SSE connection logic with reconnection
  const startSSEConnection = useCallback(() => {
    eventSourceRef.current = new EventSource(`${API_URL}/train_updates`, { withCredentials: true });

    eventSourceRef.current.onmessage = (event) => {
      dispatch({ type: 'ADD_LOG', payload: event.data });
      if (event.data.includes('Finished Training')) {
        dispatch({ type: 'SET_TRAINING_STATUS', payload: 'trained' });
        eventSourceRef.current.close();
      }
    };

    eventSourceRef.current.onerror = (error) => {
      console.error('SSE Error:', error);
      dispatch({ type: 'SET_ERROR', payload: 'Error receiving training updates' });
      eventSourceRef.current.close();

      // Attempt to reconnect after a delay
      setTimeout(() => {
        startSSEConnection();
      }, 2000);  // Adjust the reconnection delay as needed
    };
  }, []);

  const startTraining = async () => {
    dispatch({ type: 'SET_LOADING', payload: true });
    dispatch({ type: 'RESET_LOGS' });
    setInputVisible(false);
    try {
      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
        credentials: 'include'
      });
      if (!response.ok) {
        throw new Error('Failed to start training');
      }
      const data = await response.json();
      dispatch({ type: 'ADD_LOG', payload: data.message });

      if (data.message === 'Model already trained') {
        dispatch({ type: 'SET_TRAINING_STATUS', payload: 'trained' });
      } else {
        dispatch({ type: 'SET_TRAINING_STATUS', payload: 'training' });
        startSSEConnection();  // Start SSE connection
      }
    } catch (error) {
      console.error('Error:', error);
      dispatch({ type: 'SET_ERROR', payload: error.message });
    } finally {
      dispatch({ type: 'SET_LOADING', payload: false });
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      const command = inputRef.current.value.toLowerCase();
      if (command === 'start training model') {
        startTraining();
        setCommandHistory([...commandHistory, inputRef.current.value]);
        setInputVisible(false);
        inputRef.current.value = '';
      }
    }
  };

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
      <div className="terminal-container" ref={terminalRef} onScroll={(e) => e.stopPropagation()}>
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
              placeholder={state.trainingStatus === 'trained' ? "start training model or switch to canvas" : "start training model"}
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
          credentials: 'include'
        });

        if (!response.ok) {
          console.error('Failed to classify digit:', response.status, response.statusText);

          // Try to read the response body
          const responseBody = await response.text(); // Using text() to get the body content
          console.error('Response body:', responseBody);

          throw new Error(`Failed to classify digit: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        console.log(data);
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