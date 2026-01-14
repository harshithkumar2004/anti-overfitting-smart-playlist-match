# Backend Integration Guide - Anti-Overfitting Mood Classifier

## Overview

This guide provides step-by-step instructions for Node.js/Express.js backend teams to integrate the anti-overfitting mood classifier model. The model provides mood prediction and song recommendations based on audio features.

## Prerequisites

### Required Dependencies
```bash
npm install express multer python-shell fs-extra
```

### Python Dependencies
```bash
pip install scikit-learn numpy librosa pickle
```

## Model Architecture Overview

### Model Structure
The `anti_overfitting_mood_classifier.pkl` file contains:

```python
{
    'model': VotingClassifier,           # Main ensemble model
    'scaler': StandardScaler,             # Feature normalization
    'features': 15,                       # Number of input features
    'model_type': 'anti_overfitting_mood_classifier',
    'classes': ['calm', 'energetic', 'happy', 'sad'],
    'feature_names': ['feature_0', ..., 'feature_14']
}
```

### Ensemble Components
1. **GradientBoostingClassifier** (Weight: 2)
   - 50 trees, max_depth=3, learning_rate=0.05
   - Strong regularization to prevent overfitting

2. **LogisticRegression** (Weight: 1)
   - C=0.1, balanced class weights
   - Linear support for decision boundaries

## Step-by-Step Integration Guide

### Step 1: Project Setup

#### Directory Structure
```
backend/
├── server.js
├── routes/
│   └── mood.js
├── utils/
│   ├── modelLoader.js
│   ├── featureExtractor.js
│   └── recommendationEngine.js
├── models/
│   └── anti_overfitting_mood_classifier.pkl
├── uploads/
└── node_modules/
```

#### Install Dependencies
```bash
npm install express multer python-shell fs-extra path
npm install --save-dev nodemon
```

### Step 2: Model Loading Utility

#### Create `utils/modelLoader.js`
```javascript
const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs');

class ModelLoader {
    constructor() {
        this.modelPath = path.join(__dirname, '../models/anti_overfitting_mood_classifier.pkl');
        this.isLoaded = false;
        this.modelData = null;
    }

    async loadModel() {
        if (this.isLoaded) {
            return this.modelData;
        }

        try {
            const options = {
                mode: 'json',
                pythonPath: 'python',
                scriptPath: __dirname,
                args: [this.modelPath]
            };

            const result = await PythonShell.run('load_model.py', options);
            
            if (result && result[0]) {
                this.modelData = result[0];
                this.isLoaded = true;
                console.log('✅ Model loaded successfully');
                return this.modelData;
            } else {
                throw new Error('Failed to load model');
            }
        } catch (error) {
            console.error('❌ Error loading model:', error);
            throw error;
        }
    }

    getModelInfo() {
        if (!this.isLoaded) {
            throw new Error('Model not loaded. Call loadModel() first.');
        }
        
        return {
            features: this.modelData.features,
            classes: this.modelData.classes,
            model_type: this.modelData.model_type
        };
    }

    async predict(features) {
        if (!this.isLoaded) {
            await this.loadModel();
        }

        try {
            const options = {
                mode: 'json',
                pythonPath: 'python',
                scriptPath: __dirname,
                args: [
                    JSON.stringify(features),
                    this.modelPath
                ]
            };

            const result = await PythonShell.run('predict.py', options);
            
            if (result && result[0]) {
                return result[0];
            } else {
                throw new Error('Prediction failed');
            }
        } catch (error) {
            console.error('❌ Error during prediction:', error);
            throw error;
        }
    }
}

module.exports = ModelLoader;
```

#### Create `utils/load_model.py`
```python
import pickle
import sys
import json

def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Return model information
        result = {
            'features': model_data['features'],
            'classes': model_data['classes'].tolist(),
            'model_type': model_data['model_type'],
            'loaded': True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'loaded': False
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    model_path = sys.argv[1]
    load_model(model_path)
```

#### Create `utils/predict.py`
```python
import pickle
import sys
import json
import numpy as np

def predict(features_json, model_path):
    try:
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        scaler = model_data['scaler']
        classes = model_data['classes']
        
        # Parse features
        features = np.array(json.loads(features_json))
        
        # Ensure correct shape (15 features)
        if len(features) != 15:
            raise ValueError(f'Expected 15 features, got {len(features)}')
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))[0]
        
        # Make prediction
        mood = model.predict([features_scaled])[0]
        mood_probabilities = model.predict_proba([features_scaled])[0]
        confidence = float(np.max(mood_probabilities))
        
        # Create mood percentages
        mood_percentages = {}
        for i, mood_class in enumerate(classes):
            mood_percentages[mood_class] = float(mood_probabilities[i])
        
        result = {
            'mood': mood,
            'confidence': confidence,
            'mood_percentages': mood_percentages,
            'success': True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'success': False
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    features_json = sys.argv[1]
    model_path = sys.argv[2]
    predict(features_json, model_path)
```

### Step 3: Audio Feature Extraction

#### Create `utils/featureExtractor.js`
```javascript
const { PythonShell } = require('python-shell');
const path = require('path');
const fs = require('fs');

class FeatureExtractor {
    constructor() {
        this.scriptPath = __dirname;
    }

    async extractFeatures(audioFilePath) {
        try {
            const options = {
                mode: 'json',
                pythonPath: 'python',
                scriptPath: this.scriptPath,
                args: [audioFilePath]
            };

            const result = await PythonShell.run('extract_features.py', options);
            
            if (result && result[0] && result[0].success) {
                return result[0].features;
            } else {
                throw new Error(result[0]?.error || 'Feature extraction failed');
            }
        } catch (error) {
            console.error('❌ Error extracting features:', error);
            throw error;
        }
    }
}

module.exports = FeatureExtractor;
```

#### Create `utils/extract_features.py`
```python
import librosa
import numpy as np
import sys
import json

def create_discriminative_10_features(tempo_mean, energy_rms_mean, spectral_centroid_mean,
                                       zero_crossing_rate_mean, spectral_rolloff_mean,
                                       mfcc_1_mean, mfcc_2_mean, chroma_mean,
                                       onset_strength_mean, spectral_contrast_mean):
    """
    DISCRIMINATIVE 10 Features - Emotion-relevant only
    Captures rhythm, energy, timbre, brightness, harmony, and dynamics
    """
    features = np.array([
        # Core emotion features
        tempo_mean,                # Rhythm - Fast vs Slow
        energy_rms_mean,           # Energy - Loud vs Quiet
        spectral_centroid_mean,      # Brightness - Bright vs Dark
        zero_crossing_rate_mean,    # Complexity - Complex vs Simple
        
        # Advanced emotion features
        spectral_rolloff_mean,      # High frequency content (energetic vs calm)
        mfcc_1_mean,              # Timbre coefficient 1
        mfcc_2_mean,              # Timbre coefficient 2
        chroma_mean,               # Harmony - Harmonic content
        onset_strength_mean,         # Dynamics - Attack sharpness
        spectral_contrast_mean        # Spectral variation
    ])
    
    return features

def extract_discriminative_15_features(audio_path):
    """Extract 15 discriminative features - emotion-relevant only"""
    try:
        # Extract audio features directly
        y, sr = librosa.load(audio_path, duration=30)
        
        # Core emotion features
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo_mean = float(tempo)
        
        rms = librosa.feature.rms(y=y)
        energy_rms_mean = float(np.mean(rms))
        
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = float(np.mean(spectral_centroids))
        
        zcr = librosa.feature.zero_crossing_rate(y)
        zero_crossing_rate_mean = float(np.mean(zcr))
        
        # Advanced emotion features
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_rolloff_mean = float(np.mean(spectral_rolloff))
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)
        mfcc_1_mean = float(np.mean(mfcc[0]))
        mfcc_2_mean = float(np.mean(mfcc[1]))
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = float(np.mean(chroma))
        
        onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
        onset_strength_mean = float(np.mean(onset_strength))
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = float(np.mean(spectral_contrast))
        
        # Additional features for 15-feature model
        tempo_std = float(np.std(beats)) if len(beats) > 1 else 0.0
        energy_rms_std = float(np.std(rms))
        spectral_rolloff_std = float(np.std(spectral_rolloff))
        chroma_std = float(np.std(chroma))
        onset_strength_std = float(np.std(onset_strength))
        
        # Create 15-feature vector
        features = np.array([
            tempo_mean,                # Rhythm
            energy_rms_mean,           # Energy
            spectral_centroid_mean,    # Brightness
            zero_crossing_rate_mean,   # Complexity
            spectral_rolloff_mean,     # High frequency
            mfcc_1_mean,              # Timbre 1
            mfcc_2_mean,              # Timbre 2
            chroma_mean,               # Harmony
            onset_strength_mean,       # Dynamics
            spectral_contrast_mean,    # Contrast
            tempo_std,                # Tempo variability
            energy_rms_std,           # Energy variability
            spectral_rolloff_std,     # High freq variability
            chroma_std,               # Harmony variability
            onset_strength_std        # Dynamics variability
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def extract_features(audio_path):
    try:
        features = extract_discriminative_15_features(audio_path)
        
        if features is not None and len(features) == 15:
            result = {
                'success': True,
                'features': features.tolist(),
                'feature_count': len(features)
            }
        else:
            result = {
                'success': False,
                'error': 'Failed to extract correct number of features'
            }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    audio_path = sys.argv[1]
    extract_features(audio_path)
```

### Step 4: Recommendation Engine

#### Create `utils/recommendationEngine.js`
```javascript
const { PythonShell } = require('python-shell');
const path = require('path');

class RecommendationEngine {
    constructor() {
        this.scriptPath = __dirname;
        this.featureDatabase = null;
        this.audioMetadata = null;
    }

    async loadDatabase(modelPath) {
        try {
            const options = {
                mode: 'json',
                pythonPath: 'python',
                scriptPath: this.scriptPath,
                args: [modelPath]
            };

            const result = await PythonShell.run('load_database.py', options);
            
            if (result && result[0] && result[0].success) {
                this.featureDatabase = result[0].feature_database;
                this.audioMetadata = result[0].audio_metadata;
                console.log('✅ Database loaded successfully');
                return true;
            } else {
                throw new Error(result[0]?.error || 'Database loading failed');
            }
        } catch (error) {
            console.error('❌ Error loading database:', error);
            throw error;
        }
    }

    async findSimilarSongs(queryFeatures, topK = 5) {
        if (!this.featureDatabase) {
            throw new Error('Database not loaded. Call loadDatabase() first.');
        }

        try {
            const options = {
                mode: 'json',
                pythonPath: 'python',
                scriptPath: this.scriptPath,
                args: [
                    JSON.stringify(queryFeatures),
                    JSON.stringify(this.featureDatabase),
                    JSON.stringify(this.audioMetadata),
                    topK.toString()
                ]
            };

            const result = await PythonShell.run('find_similar.py', options);
            
            if (result && result[0] && result[0].success) {
                return result[0].recommendations;
            } else {
                throw new Error(result[0]?.error || 'Similarity search failed');
            }
        } catch (error) {
            console.error('❌ Error finding similar songs:', error);
            throw error;
        }
    }
}

module.exports = RecommendationEngine;
```

#### Create `utils/load_database.py`
```python
import pickle
import sys
import json
import numpy as np
from pathlib import Path

def load_database(model_path):
    try:
        # Load model data
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # For this example, we'll create a mock database
        # In production, you would load your actual feature database
        num_songs = 1744
        num_features = 15
        
        # Mock feature database (replace with actual database)
        feature_database = np.random.rand(num_songs, num_features)
        
        # Mock audio metadata
        audio_metadata = []
        for i in range(num_songs):
            audio_metadata.append({
                'filename': f'song_{i:04d}.mp3',
                'song_id': i,
                'audio_path': f'/audio/song_{i:04d}.mp3'
            })
        
        result = {
            'success': True,
            'feature_database': feature_database.tolist(),
            'audio_metadata': audio_metadata,
            'database_size': num_songs
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    model_path = sys.argv[1]
    load_database(model_path)
```

#### Create `utils/find_similar.py`
```python
import numpy as np
import sys
import json
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_songs(query_features_json, feature_database_json, audio_metadata_json, top_k=5):
    try:
        # Parse inputs
        query_features = np.array(json.loads(query_features_json))
        feature_database = np.array(json.loads(feature_database_json))
        audio_metadata = json.loads(audio_metadata_json)
        top_k = int(top_k)
        
        # Calculate similarity metrics
        cosine_similarities = cosine_similarity([query_features], feature_database)[0]
        euclidean_distances = np.linalg.norm(feature_database - query_features, axis=1)
        
        # Normalize Euclidean distances to similarity scores (0-1)
        max_dist = np.max(euclidean_distances)
        euclidean_similarities = 1 - (euclidean_distances / max_dist)
        
        # Combine similarities with weights (cosine more important)
        combined_similarities = 0.7 * cosine_similarities + 0.3 * euclidean_similarities
        
        # Get top-k most similar songs
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            song_data = audio_metadata[idx]
            
            # Extract features for display (simplified)
            features = feature_database[idx]
            tempo_mean = float(features[0])
            energy_mean = float(features[1])
            
            recommendations.append({
                'filename': song_data['filename'],
                'similarity_score': float(combined_similarities[idx]),
                'tempo': {
                    'mean_bpm': round(tempo_mean, 2),
                    'variability': 10.0  # Simplified
                },
                'loudness': {
                    'mean_energy': round(energy_mean, 4),
                    'variability': 0.05  # Simplified
                },
                'audio_path': song_data['audio_path']
            })
        
        result = {
            'success': True,
            'recommendations': recommendations,
            'query_processed': True
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e)
        }
        print(json.dumps(error_result))

if __name__ == "__main__":
    query_features_json = sys.argv[1]
    feature_database_json = sys.argv[2]
    audio_metadata_json = sys.argv[3]
    top_k = sys.argv[4]
    find_similar_songs(query_features_json, feature_database_json, audio_metadata_json, top_k)
```

### Step 5: Express.js Routes

#### Create `routes/mood.js`
```javascript
const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const ModelLoader = require('../utils/modelLoader');
const FeatureExtractor = require('../utils/featureExtractor');
const RecommendationEngine = require('../utils/recommendationEngine');

const router = express.Router();

// Initialize components
const modelLoader = new ModelLoader();
const featureExtractor = new FeatureExtractor();
const recommendationEngine = new RecommendationEngine();

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ 
    storage: storage,
    fileFilter: (req, file, cb) => {
        if (file.mimetype.startsWith('audio/')) {
            cb(null, true);
        } else {
            cb(new Error('Only audio files are allowed'), false);
        }
    },
    limits: {
        fileSize: 10 * 1024 * 1024 // 10MB limit
    }
});

// Initialize model and database on startup
async function initializeServices() {
    try {
        await modelLoader.loadModel();
        await recommendationEngine.loadDatabase(
            path.join(__dirname, '../models/anti_overfitting_mood_classifier.pkl')
        );
        console.log('✅ All services initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize services:', error);
    }
}

// Initialize on module load
initializeServices();

// POST /api/mood/predict - Predict mood from uploaded audio
router.post('/predict', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({
                success: false,
                error: 'No audio file uploaded'
            });
        }

        console.log('🔧 Extracting features from:', req.file.path);
        
        // Extract features
        const features = await featureExtractor.extractFeatures(req.file.path);
        
        console.log('🎯 Making prediction...');
        
        // Make prediction
        const prediction = await modelLoader.predict(features);
        
        console.log('🔍 Finding similar songs...');
        
        // Find similar songs
        const recommendations = await recommendationEngine.findSimilarSongs(features, 5);
        
        // Clean up uploaded file
        fs.unlinkSync(req.file.path);
        
        // Return response
        res.json({
            success: true,
            mood: prediction.mood,
            confidence: prediction.confidence,
            mood_percentages: prediction.mood_percentages,
            recommendations: recommendations,
            features_used: features.length
        });
        
    } catch (error) {
        console.error('❌ Error in prediction:', error);
        
        // Clean up uploaded file if it exists
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
        
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// GET /api/mood/info - Get model information
router.get('/info', async (req, res) => {
    try {
        const modelInfo = modelLoader.getModelInfo();
        
        res.json({
            success: true,
            model: modelInfo,
            status: 'ready'
        });
        
    } catch (error) {
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

// POST /api/mood/features - Predict from pre-extracted features
router.post('/features', async (req, res) => {
    try {
        const { features } = req.body;
        
        if (!features || !Array.isArray(features) || features.length !== 15) {
            return res.status(400).json({
                success: false,
                error: 'Invalid features. Expected array of 15 numbers.'
            });
        }
        
        // Make prediction
        const prediction = await modelLoader.predict(features);
        
        // Find similar songs
        const recommendations = await recommendationEngine.findSimilarSongs(features, 5);
        
        res.json({
            success: true,
            mood: prediction.mood,
            confidence: prediction.confidence,
            mood_percentages: prediction.mood_percentages,
            recommendations: recommendations
        });
        
    } catch (error) {
        console.error('❌ Error in feature prediction:', error);
        res.status(500).json({
            success: false,
            error: error.message
        });
    }
});

module.exports = router;
```

### Step 6: Main Server File

#### Create `server.js`
```javascript
const express = require('express');
const cors = require('cors');
const moodRoutes = require('./routes/mood');
const path = require('path');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir);
}

// Routes
app.use('/api/mood', moodRoutes);

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'anti-overfitting-mood-classifier'
    });
});

// Serve static audio files (if you have them)
app.use('/audio', express.static(path.join(__dirname, 'audio')));

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('❌ Server error:', error);
    
    if (error.code === 'LIMIT_FILE_SIZE') {
        return res.status(413).json({
            success: false,
            error: 'File too large. Maximum size is 10MB.'
        });
    }
    
    res.status(500).json({
        success: false,
        error: 'Internal server error'
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint not found'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`🚀 Anti-Overfitting Mood Classifier API running on port ${PORT}`);
    console.log(`📡 Health check: http://localhost:${PORT}/health`);
    console.log(`🎵 Mood prediction: http://localhost:${PORT}/api/mood/predict`);
    console.log(`📊 Model info: http://localhost:${PORT}/api/mood/info`);
});

module.exports = app;
```

### Step 7: Package Configuration

#### Create `package.json`
```json
{
  "name": "anti-overfitting-mood-classifier-backend",
  "version": "1.0.0",
  "description": "Backend API for anti-overfitting mood classification",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5-lts.1",
    "python-shell": "^5.0.0",
    "cors": "^2.8.5",
    "fs-extra": "^11.1.1",
    "path": "^0.12.7"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  },
  "keywords": [
    "mood-classification",
    "machine-learning",
    "audio-processing",
    "anti-overfitting"
  ],
  "author": "Backend Team",
  "license": "MIT"
}
```

## API Endpoints

### 1. POST /api/mood/predict
**Purpose**: Predict mood from uploaded audio file

**Request**: 
- Method: POST
- Content-Type: multipart/form-data
- Body: audio file (MP3, WAV, etc.)

**Response**:
```json
{
  "success": true,
  "mood": "happy",
  "confidence": 0.75,
  "mood_percentages": {
    "happy": 0.75,
    "sad": 0.15,
    "calm": 0.05,
    "energetic": 0.05
  },
  "recommendations": [
    {
      "filename": "song_0001.mp3",
      "similarity_score": 0.85,
      "tempo": {
        "mean_bpm": 120.5,
        "variability": 10.2
      },
      "loudness": {
        "mean_energy": 0.6543,
        "variability": 0.0123
      },
      "audio_path": "/audio/song_0001.mp3"
    }
  ],
  "features_used": 15
}
```

### 2. POST /api/mood/features
**Purpose**: Predict mood from pre-extracted features

**Request**:
```json
{
  "features": [120.5, 0.6543, 2000.1, 0.1234, 2100.5, -100.2, -20.1, 0.5432, 0.1234, 20.1, 15.2, 0.0543, 100.2, 0.1234, 0.0543]
}
```

**Response**: Same as /api/mood/predict

### 3. GET /api/mood/info
**Purpose**: Get model information

**Response**:
```json
{
  "success": true,
  "model": {
    "features": 15,
    "classes": ["calm", "energetic", "happy", "sad"],
    "model_type": "anti_overfitting_mood_classifier"
  },
  "status": "ready"
}
```

## Deployment Instructions

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/harshithkumar2004/anti-overfitting-smart-playlist-match.git
cd anti-overfitting-smart-playlist-match

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install scikit-learn numpy librosa

# Create required directories
mkdir uploads
mkdir audio
```

### 2. Place Model File
```bash
# Copy the model file to the models directory
cp anti_overfitting_mood_classifier.pkl backend/models/
```

### 3. Start Server
```bash
# Development mode
npm run dev

# Production mode
npm start
```

### 4. Test API
```bash
# Health check
curl http://localhost:3000/health

# Model info
curl http://localhost:3000/api/mood/info

# Upload audio file
curl -X POST -F "audio=@test_song.mp3" http://localhost:3000/api/mood/predict
```

## Error Handling

### Common Errors and Solutions

#### 1. Model Loading Error
**Error**: `Failed to load model`
**Solution**: 
- Verify model file path
- Check Python dependencies
- Ensure model file is not corrupted

#### 2. Feature Extraction Error
**Error**: `Failed to extract correct number of features`
**Solution**:
- Verify audio file format (MP3, WAV)
- Check audio file duration (should be > 30 seconds)
- Ensure librosa is installed correctly

#### 3. Prediction Error
**Error**: `Expected 15 features, got X`
**Solution**:
- Verify feature extraction pipeline
- Check feature array length
- Ensure consistent feature order

#### 4. Memory Issues
**Error**: `Memory allocation failed`
**Solution**:
- Increase server memory
- Optimize feature database loading
- Use streaming for large files

## Performance Optimization

### 1. Caching
- Cache model predictions for repeated requests
- Store feature database in memory
- Use Redis for distributed caching

### 2. Async Processing
- Use worker threads for feature extraction
- Implement request queuing for high load
- Use streaming for large file uploads

### 3. Database Optimization
- Index feature vectors for faster similarity search
- Use approximate nearest neighbor algorithms
- Implement pagination for large databases

## Security Considerations

### 1. File Upload Security
- Validate file types and sizes
- Scan uploaded files for malware
- Use secure file storage

### 2. API Security
- Implement rate limiting
- Use API keys for authentication
- Validate input data

### 3. Model Security
- Protect model file from unauthorized access
- Monitor for adversarial inputs
- Log prediction requests for auditing

## Monitoring and Logging

### 1. Health Monitoring
- Monitor API response times
- Track prediction accuracy
- Log system resource usage

### 2. Error Tracking
- Log all prediction errors
- Monitor feature extraction failures
- Track database connection issues

### 3. Performance Metrics
- Track request throughput
- Monitor memory usage
- Measure prediction latency

This comprehensive guide provides everything needed to integrate the anti-overfitting mood classifier into a Node.js/Express.js backend with proper error handling, security, and performance considerations.
