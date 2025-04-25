<<<<<<< HEAD
# DeepFake Detection Platform

A full-stack web application for detecting deepfake videos using AI/ML. The platform allows users to upload videos and analyze them for potential deepfake manipulation.

## Features

- User authentication (login/signup)
- Video upload with drag-and-drop support
- Real-time deepfake detection analysis
- Modern, responsive UI with animations
- Secure API endpoints
- MongoDB database integration

## Tech Stack

- Frontend: React, React Router, Framer Motion, TailwindCSS
- Backend: Node.js, Express
- Database: MongoDB
- Authentication: JWT, bcrypt
- File Upload: Multer

## Project Structure

```
├── frontend/           # React frontend application
│   ├── src/           # Source code
│   ├── public/        # Public assets
│   └── package.json   # Frontend dependencies
│
├── backend/           # Node.js backend server
│   ├── routes/       # API routes
│   ├── middleware/   # Custom middleware
│   ├── models/       # Database models
│   ├── ml/          # ML model integration
│   └── uploads/      # File uploads directory
│
├── ml/               # Machine Learning components
│   ├── model.py      # Core ML model definition
│   ├── train_model.py # Training script
│   └── preprocess.py # Data preprocessing utilities
│
├── tests/            # Testing files
│   ├── test_model.py # Model tests
│   ├── inspect_model.py # Model inspection utilities
│   └── test_videos/  # Test video files
│
├── scripts/          # Utility scripts
│   └── download_test_videos.py # Video download utility
│
├── assets/           # Static assets
│   └── image.png
│
└── logs/            # Application logs

```

## Setup and Installation

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Backend Setup
```bash
cd backend
npm install
node server.js
```

### ML Setup
```bash
cd ml
pip install -r requirements.txt
python train_model.py
```

## Development

### Running Tests
```bash
cd tests
python test_model.py
```

### Scripts
- `scripts/download_test_videos.py`: Utility to download test videos
- `ml/train_model.py`: Train the ML model
- `ml/preprocess.py`: Preprocess data for training

## Directory Structure Explanation

- `frontend/`: Contains the React.js frontend application
- `backend/`: Contains the Node.js backend server
- `ml/`: Contains all machine learning related code
- `tests/`: Contains all test files and test data
- `scripts/`: Contains utility scripts
- `assets/`: Contains static assets
- `logs/`: Contains application logs

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

[Add your license here]

## API Endpoints

- POST /api/auth/register - User registration
- POST /api/auth/login - User login
- POST /api/videos/upload - Video upload
- GET /api/videos/analysis/:id - Get video analysis results

## Security Features

- JWT-based authentication
- Password hashing with bcrypt
- Input validation and sanitization
- Protected API routes
- Secure file upload handling 
=======
# DEEPFAKE-DETECTION-MODEL
>>>>>>> ce9d661b7842f07871c2f4c32f2fa740395bcfcf
