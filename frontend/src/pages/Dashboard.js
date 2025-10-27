import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import TestVideo from '../components/TestVideo';

const Dashboard = () => {
  const [videos, setVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [showUpload, setShowUpload] = useState(false);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    onDrop: handleUpload,
  });

  useEffect(() => {
    // Load any existing videos from localStorage or API
    const savedVideos = localStorage.getItem('uploadedVideos');
    if (savedVideos) {
      setVideos(JSON.parse(savedVideos));
    }
  }, []);

  async function handleUpload(acceptedFiles) {
    const file = acceptedFiles[0];
    if (!file) return;

    setUploading(true);
    setError('');

    const formData = new FormData();
    formData.append('video', file);

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/videos/upload`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      const newVideos = [response.data, ...videos];
      setVideos(newVideos);
      localStorage.setItem('uploadedVideos', JSON.stringify(newVideos));
    } catch (error) {
      setError(error.response?.data?.message || 'Error uploading video');
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 py-12">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-3xl font-extrabold text-white sm:text-4xl">
            Deepfake Detection Dashboard
          </h1>
          <p className="mt-3 text-lg text-gray-300">
            Upload and test videos for deepfake detection
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Upload Section - Only show when no video is selected */}
          {!showUpload && (
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ delay: 0.2 }}
              className="bg-gray-800 rounded-lg p-6 shadow-lg"
            >
              <h2 className="text-2xl font-bold text-white mb-4">Upload Video</h2>
              
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                  isDragActive
                    ? 'border-blue-400 bg-blue-900/20'
                    : 'border-gray-600 hover:border-gray-500'
                }`}
              >
                <input {...getInputProps()} />
                <div className="space-y-4">
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    stroke="currentColor"
                    fill="none"
                    viewBox="0 0 48 48"
                  >
                    <path
                      d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02"
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                  <div>
                    <p className="text-lg text-gray-300">
                      {isDragActive
                        ? 'Drop the video here...'
                        : 'Drag & drop a video here, or click to select'}
                    </p>
                    <p className="text-sm text-gray-400 mt-2">
                      Supports MP4, MOV, AVI, MKV, WebM (max 100MB)
                    </p>
                  </div>
                </div>
              </div>

              {uploading && (
                <div className="mt-4 flex items-center justify-center">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
                  <span className="ml-2 text-gray-300">Uploading...</span>
                </div>
              )}

              {error && (
                <div className="mt-4 text-red-500 text-sm bg-red-900/20 p-3 rounded-md">
                  {error}
                </div>
              )}
            </motion.div>
          )}

          {/* Test Video Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
            className={showUpload ? "lg:col-span-2" : ""}
          >
            <TestVideo onVideoSelect={() => setShowUpload(true)} onVideoClear={() => setShowUpload(false)} />
          </motion.div>
        </div>

        {/* Recent Videos Section */}
        {videos.length > 0 && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
            className="mt-12"
          >
            <h2 className="text-2xl font-bold text-white mb-6">Recent Videos</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {videos.map((video, index) => (
                <div key={index} className="bg-gray-800 rounded-lg p-4 shadow-lg">
                  <div className="aspect-video bg-gray-700 rounded-lg mb-4 flex items-center justify-center">
                    <svg className="h-12 w-12 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                    </svg>
                  </div>
                  <h3 className="text-white font-medium truncate">{video.name || `Video ${index + 1}`}</h3>
                  <p className="text-gray-400 text-sm">
                    {video.size ? `${(video.size / 1024 / 1024).toFixed(1)} MB` : 'Unknown size'}
                  </p>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Dashboard; 