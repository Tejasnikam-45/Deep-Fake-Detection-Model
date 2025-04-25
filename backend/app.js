const express = require('express');
const cors = require('cors');
const path = require('path');
const videosRouter = require('./routes/videos');

const app = express();

// Enable CORS for all routes
app.use(cors({
    origin: 'http://localhost:3000',
    credentials: true
}));

// Parse JSON bodies
app.use(express.json());

// Serve static files from the test_results directory
app.use('/results', express.static(path.join(__dirname, 'ml/test_results')));

// Routes
app.use('/api/videos', videosRouter);

// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).json({ 
        error: 'Something broke!',
        details: err.message 
    });
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
}); 