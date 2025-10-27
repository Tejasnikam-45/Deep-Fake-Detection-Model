import React, { useState } from 'react';
import axios from 'axios';
import { motion } from 'framer-motion';

const TestVideo = ({ onVideoSelect, onVideoClear }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [verdict, setVerdict] = useState(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setError(null);
    setVerdict(null);
    setResults(null);
    
    // Notify parent component that a video has been selected
    if (selectedFile && onVideoSelect) {
      onVideoSelect();
    }
  };

  const calculateVerdict = (predictions) => {
    if (!predictions || predictions.length === 0) {
      console.log("No predictions available");
      return null;
    }

    console.log("Raw predictions:", predictions);
    
    // If predictions is already the verdict object
    if (predictions.verdict) {
      console.log("Using direct verdict:", predictions.verdict);
      return {
        label: predictions.verdict,
        confidence: (predictions.score * 100).toFixed(1)
      };
    }

    // If predictions is an array, calculate average
    const avgPrediction = Array.isArray(predictions) 
      ? predictions.reduce((a, b) => a + b, 0) / predictions.length
      : predictions;
    
    console.log("Average prediction:", avgPrediction);
    
    return {
      label: avgPrediction > 0.5 ? 'FAKE' : 'REAL',
      confidence: (Math.abs(avgPrediction - 0.5) * 200).toFixed(1)
    };
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      setError('Please select a video file');
      return;
    }

    setLoading(true);
    setError(null);
    setVerdict(null);
    setResults(null);

    const formData = new FormData();
    formData.append('video', file);

    try {
      console.log('Sending video for analysis...');
      const response = await axios.post(`${process.env.REACT_APP_API_URL}/api/videos/test`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        withCredentials: true
      });

      console.log('Server response:', response.data);

      if (response.data.error) {
        setError(response.data.error);
        return;
      }

      const results = response.data.results;
      console.log('Processing results:', results);
      
      if (!results) {
        setError('No results received from the server');
        return;
      }

      // Set verdict first
      setVerdict({
        label: results.verdict,
        score: results.score
      });
      
      // Then set results with sample faces
      if (results.sampleFaces && results.sampleFaces.length > 0) {
        console.log(`Received ${results.sampleFaces.length} sample faces`);
        setResults({
          sampleFaces: results.sampleFaces,
          predictions: results.predictions || []
        });
      } else {
        console.log('No sample faces received from server');
      }
      
      console.log('Verdict set:', results.verdict, 'Score:', results.score);
    } catch (err) {
      console.error('Error details:', err);
      setError(err.response?.data?.error || 'Error processing video. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 shadow-lg">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-white">Test Video</h2>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-gray-400">AI Ready</span>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Select Video File
          </label>
          <div className="relative">
            <input
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="block w-full text-sm text-gray-300
                file:mr-4 file:py-3 file:px-6
                file:rounded-lg file:border-0
                file:text-sm file:font-semibold
                file:bg-gradient-to-r file:from-blue-500 file:to-purple-600 
                file:text-white file:shadow-lg
                hover:file:from-blue-600 hover:file:to-purple-700
                transition-all duration-200"
            />
            <div className="absolute inset-y-0 right-0 flex items-center pr-3 pointer-events-none">
              <svg className="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
            </div>
          </div>
          {file && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-3 p-3 bg-gray-700 rounded-lg border border-gray-600"
            >
              <div className="flex items-center space-x-3">
                <svg className="h-8 w-8 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                </svg>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-white truncate">{file.name}</p>
                  <p className="text-xs text-gray-400">
                    {(file.size / 1024 / 1024).toFixed(1)} MB
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setFile(null);
                    setError(null);
                    setVerdict(null);
                    setResults(null);
                    
                    // Notify parent component that video has been cleared
                    if (onVideoClear) {
                      onVideoClear();
                    }
                  }}
                  className="text-gray-400 hover:text-white transition-colors"
                >
                  <svg className="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </motion.div>
          )}
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-red-400 text-sm bg-red-900/20 border border-red-500/30 p-4 rounded-lg"
          >
            <div className="flex items-start space-x-2">
              <svg className="h-5 w-5 text-red-400 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="font-medium">{error}</p>
                {error.includes('backend server') && (
                  <div className="mt-2 text-xs space-y-1">
                    <p>Please ensure that:</p>
                    <ul className="list-disc list-inside space-y-1 ml-2">
                      <li>The backend server is running (cd backend && npm start)</li>
                      <li>The .env file has the correct API URL</li>
                      <li>The uploads directory exists in the backend</li>
                    </ul>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}

        <button
          type="submit"
          disabled={loading || !file}
          className={`w-full py-3 px-6 rounded-lg text-white font-semibold text-lg transition-all duration-200 transform
            ${loading || !file
              ? 'bg-gray-600 cursor-not-allowed scale-95'
              : 'bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 hover:scale-105 shadow-lg hover:shadow-xl'
            }`}
        >
          {loading ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Analyzing Video...
            </span>
          ) : (
            <span className="flex items-center justify-center">
              <svg className="mr-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
              Test for Deepfake
            </span>
          )}
        </button>
      </form>

      {verdict && (
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8"
        >
          <div className="text-center">
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-300 mb-2">Analysis Result</h3>
              <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-purple-600 mx-auto rounded-full"></div>
            </div>
            
            <div 
              className={`inline-block px-8 py-6 rounded-2xl shadow-2xl transform transition-all duration-300 hover:scale-105 ${
                verdict.label === 'FAKE' 
                  ? 'bg-gradient-to-br from-red-500 to-red-700 border-2 border-red-400' 
                  : 'bg-gradient-to-br from-green-500 to-green-700 border-2 border-green-400'
              }`}
              style={{
                boxShadow: verdict.label === 'FAKE' 
                  ? '0 0 30px rgba(239, 68, 68, 0.5)' 
                  : '0 0 30px rgba(34, 197, 94, 0.5)'
              }}
            >
              <div className="flex items-center justify-center space-x-3">
                {verdict.label === 'FAKE' ? (
                  <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                ) : (
                  <svg className="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
                <h3 
                  className="text-5xl font-black text-white"
                  style={{
                    textShadow: '2px 2px 4px rgba(0,0,0,0.6)',
                    letterSpacing: '0.1em'
                  }}
                >
                  {verdict.label}
                </h3>
              </div>
            </div>

            {/* Display Performance Metrics */}
            {results && results.metrics && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="mt-8 grid grid-cols-2 gap-4"
              >
                <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-lg border border-gray-600">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Precision</h4>
                  <p className="text-2xl font-bold text-blue-400">{results.metrics.precision.toFixed(2)}%</p>
                </div>
                <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-lg border border-gray-600">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Recall</h4>
                  <p className="text-2xl font-bold text-green-400">{results.metrics.recall.toFixed(2)}%</p>
                </div>
                <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-lg border border-gray-600">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">F1 Score</h4>
                  <p className="text-2xl font-bold text-purple-400">{results.metrics.f1_score.toFixed(2)}%</p>
                </div>
                <div className="bg-gray-700/50 backdrop-blur-sm p-4 rounded-lg border border-gray-600">
                  <h4 className="text-sm font-medium text-gray-300 mb-2">Accuracy</h4>
                  <p className="text-2xl font-bold text-yellow-400">{results.metrics.accuracy.toFixed(2)}%</p>
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
      )}

      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8"
        >
          <div className="mb-6">
            <h3 className="text-xl font-semibold text-white mb-2">Detailed Analysis</h3>
            <div className="w-16 h-1 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full"></div>
          </div>
          
          {/* Display sample faces */}
          {results.sampleFaces && results.sampleFaces.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="mb-8"
            >
              <h4 className="text-lg font-medium text-white mb-4 flex items-center">
                <svg className="h-5 w-5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                </svg>
                Sample Faces from Video
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {results.sampleFaces.map((face, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.1 }}
                    className="relative group"
                  >
                    <div className="relative overflow-hidden rounded-lg shadow-lg">
                      <img
                        src={`data:image/jpeg;base64,${face}`}
                        alt={`Face ${index + 1}`}
                        className="w-full h-32 object-cover transition-transform duration-300 group-hover:scale-110"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                      <div className="absolute bottom-2 left-2 right-2 text-white text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                        Face {index + 1}
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}
          
          {/* Display prediction plot */}
          {results.plot && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="mb-6"
            >
              <h4 className="text-lg font-medium text-white mb-4 flex items-center">
                <svg className="h-5 w-5 mr-2 text-purple-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
                Prediction Analysis
              </h4>
              <div className="bg-gray-700/50 backdrop-blur-sm rounded-lg p-4 border border-gray-600">
                <img
                  src={`data:image/png;base64,${results.plot}`}
                  alt="Prediction Plot"
                  className="w-full rounded-lg shadow-lg"
                />
              </div>
            </motion.div>
          )}

          {/* Analysis Summary */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg p-6 border border-gray-600"
          >
            <h4 className="text-lg font-medium text-white mb-4 flex items-center">
              <svg className="h-5 w-5 mr-2 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Analysis Summary
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div className="space-y-2">
                <p className="text-gray-300">
                  <span className="font-medium text-white">Total Frames Analyzed:</span> {results.sampleFaces?.length || 'N/A'}
                </p>
                <p className="text-gray-300">
                  <span className="font-medium text-white">Detection Method:</span> AI-Powered Deep Learning
                </p>
              </div>
              <div className="space-y-2">
                <p className="text-gray-300">
                  <span className="font-medium text-white">Model Version:</span> v2.1.0
                </p>
                <p className="text-gray-300">
                  <span className="font-medium text-white">Processing Time:</span> Real-time
                </p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
};

export default TestVideo; 