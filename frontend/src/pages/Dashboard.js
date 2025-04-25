import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import TestVideo from '../components/TestVideo';

const Dashboard = () => {
  const [videos, setVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const { user } = useAuth();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.mov', '.avi'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    onDrop: handleUpload,
  });

  useEffect(() => {
    fetchVideos();
  }, []);

  const fetchVideos = async () => {
    try {
      const response = await axios.get(`${process.env.REACT_APP_API_URL}/api/videos/my-videos`);
      setVideos(response.data);
    } catch (error) {
      console.error('Error fetching videos:', error);
    }
  };

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
      setVideos([response.data, ...videos]);
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
            Test videos for deepfake detection
          </p>
        </motion.div>

        <div className="max-w-3xl mx-auto">
          {/* Test Video Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <TestVideo />
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 