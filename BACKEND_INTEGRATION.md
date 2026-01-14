# Backend Integration Guide - Anti-Overfitting Mood Classifier

## Overview

This guide explains how to integrate the anti-overfitting mood classifier model into a Node.js/Express.js backend. The model provides both mood prediction and song recommendations based on audio feature similarity.

## How It Works

### Model Architecture
The `.pkl` file contains a complete machine learning pipeline:
- **Model**: VotingClassifier (Gradient Boosting + Logistic Regression)
- **Scaler**: StandardScaler for feature normalization
- **Feature Database**: Pre-computed features for 1,744 songs
- **Metadata**: Song information and feature mappings

### Prediction Flow
1. **Audio Upload** → Feature Extraction (15 audio features)
2. **Feature Scaling** → StandardScaler normalization
3. **Mood Prediction** → VotingClassifier prediction
4. **Similarity Search** → Find 5 most similar songs from database
5. **Response** → Mood + 5 recommendations with similarity scores

## Node.js Integration Requirements

### Dependencies
```json
{
  "dependencies": {
    "express": "^4.18.2",
    "multer": "^1.4.5-lts.1",
    "python-shell": "^5.0.0",
    "cors": "^2.8.5",
    "dotenv": "^16.3.1"
  }
}
```

### Python Dependencies (for Python bridge)
```bash
pip install scikit-learn==1.3.0 numpy==1.24.3 librosa==0.10.1
```

## Implementation Options

### Option 1: Python Bridge (Recommended)
Use Python shell to execute Python code from Node.js

### Option 2: Python Microservice
Run Python as separate API service

### Option 3: Model Conversion
Convert to ONNX/TF.js format (complex, not recommended)

---

## Option 1: Python Bridge Implementation

### Step 1: Create Python Prediction Script

Create `predict.py` in your backend:

```python
import pickle
import numpy as np
import librosa
import sys
import json
from pathlib import Path

# Load model and database
MODEL_PATH = Path(__file__).parent / 'anti_overfitting_mood_classifier.pkl'

def load_model():
    """Load the anti-overfitting mood classifier"""
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    
    return model_data

def extract_features(audio_path):
    """Extract 15 discriminative features from audio file"""
    try:
        # Load audio
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
        
        # Additional features for 15 total
        tempo_std = float(np.std(beats)) if len(beats) > 1 else 0.0
        energy_rms_std = float(np.std(rms))
        spectral_rolloff_std = float(np.std(spectral_rolloff))
        chroma_std = float(np.std(chroma))
        onset_strength_std = float(np.std(onset_strength))
        
        # Create feature array (15 features)
        features = np.array([
            tempo_mean, energy_rms_mean, spectral_centroid_mean,
            zero_crossing_rate_mean, spectral_rolloff_mean,
            mfcc_1_mean, mfcc_2_mean, chroma_mean,
            onset_strength_mean, spectral_contrast_mean,
            tempo_std, energy_rms_std, spectral_rolloff_std,
            chroma_std, onset_strength_std
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_mood_and_recommendations(audio_path):
    """Predict mood and get recommendations"""
    try:
        # Load model
        model_data = load_model()
        model = model_data['model']
        scaler = model_data['scaler']
        feature_database = model_data.get('feature_database')
        audio_metadata = model_data.get('audio_metadata')
        
        # Extract features
        features = extract_features(audio_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))[0]
        
        # Predict mood
        mood = model.predict([features_scaled])[0]
        mood_probabilities = model.predict_proba([features_scaled])[0]
        confidence = float(np.max(mood_probabilities))
        
        # Get all mood percentages
        mood_classes = model.classes_
        mood_percentages = {}
        for i, mood_class in enumerate(mood_classes):
            mood_percentages[mood_class] = float(mood_probabilities[i])
        
        # Find similar songs
        recommendations = []
        if feature_database is not None and audio_metadata is not None:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Calculate similarities
            similarities = cosine_similarity([features_scaled], feature_database)[0]
            
            # Get top 5 most similar
            top_indices = np.argsort(similarities)[::-1][:5]
            
            for idx in top_indices:
                song_data = audio_metadata.iloc[idx].to_dict()
                similarity_score = float(similarities[idx])
                
                # Extract display features
                original_features = scaler.inverse_transform([feature_database[idx]])[0]
                tempo_mean = float(original_features[0]) if len(original_features) >= 1 else 120
                energy_mean = float(original_features[1]) if len(original_features) >= 2 else 0.5
                
                recommendations.append({
                    'filename': song_data.get('filename', f'song_{idx}.mp3'),
                    'similarity_score': similarity_score,
                    'tempo': round(tempo_mean, 2),
                    'energy': round(energy_mean, 4),
                    'audio_path': f'/audio/{song_data.get("filename", f"song_{idx}.mp3")}'
                })
        
        # Prepare response
        response = {
            'mood': mood,
            'confidence': confidence,
            'mood_percentages': mood_percentages,
            'recommendations': recommendations,
            'features_extracted': len(features)
        }
        
        return response
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

if __name__ == "__main__":
    # Command line interface
    if len(sys.argv) != 2:
        print("Usage: python predict.py <audio_file_path>")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = predict_mood_and_recommendations(audio_path)
    
    if result:
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps({'error': 'Prediction failed'}))
        sys.exit(1)
```

### Step 2: Create Express.js Backend

Create `server.js`:

```javascript
const express = require('express');
const multer = require('multer');
const { PythonShell } = require('python-shell');
const cors = require('cors');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Configure multer for file uploads
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        cb(null, Date.now() + '-' + file.originalname);
    }
});

const upload = multer({ storage: storage });

// Ensure uploads directory exists
async function ensureUploadsDir() {
    try {
        await fs.mkdir('uploads', { recursive: true });
    } catch (error) {
        console.error('Error creating uploads directory:', error);
    }
}

// Prediction endpoint
app.post('/predict', upload.single('audio'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'No audio file uploaded' });
        }

        const audioPath = req.file.path;
        
        // Call Python script for prediction
        const options = {
            mode: 'json',
            pythonPath: 'python', // or 'python3' depending on your system
            scriptPath: __dirname,
            args: [audioPath]
        };

        PythonShell.run('predict.py', options, (err, results) => {
            // Clean up uploaded file
            fs.unlink(audioPath).catch(console.error);

            if (err) {
                console.error('Python script error:', err);
                return res.status(500).json({ error: 'Prediction failed' });
            }

            if (results && results.length > 0) {
                const result = results[0];
                
                // Validate result structure
                if (result.error) {
                    return res.status(500).json({ error: result.error });
                }

                // Format response
                const response = {
                    mood: result.mood,
                    confidence: result.confidence,
                    mood_percentages: result.mood_percentages,
                    recommendations: result.recommendations || [],
                    features_extracted: result.features_extracted
                };

                res.json(response);
            } else {
                res.status(500).json({ error: 'No results from prediction' });
            }
        });

    } catch (error) {
        console.error('Server error:', error);
        res.status(500).json({ error: 'Internal server error' });
    }
});

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy', 
        model: 'anti_overfitting_mood_classifier',
        version: '1.0'
    });
});

// Serve static audio files
app.use('/audio', express.static('audio'));

// Start server
async function startServer() {
    await ensureUploadsDir();
    
    app.listen(PORT, () => {
        console.log(`Server running on port ${PORT}`);
        console.log(`Model endpoint: http://localhost:${PORT}/predict`);
        console.log(`Health check: http://localhost:${PORT}/health`);
    });
}

startServer().catch(console.error);
```

### Step 3: Package.json

```json
{
  "name": "anti-overfitting-mood-classifier-backend",
  "version": "1.0.0",
  "description": "Backend API for anti-overfitting mood classifier",
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
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "nodemon": "^3.0.1"
  }
}
```

## API Response Format

### Successful Response
```json
{
  "mood": "happy",
  "confidence": 0.73,
  "mood_percentages": {
    "happy": 0.73,
    "sad": 0.12,
    "calm": 0.08,
    "energetic": 0.07
  },
  "recommendations": [
    {
      "filename": "song_123.mp3",
      "similarity_score": 0.89,
      "tempo": 128.5,
      "energy": 0.6234,
      "audio_path": "/audio/song_123.mp3"
    },
    {
      "filename": "song_456.mp3",
      "similarity_score": 0.85,
      "tempo": 125.2,
      "energy": 0.5891,
      "audio_path": "/audio/song_456.mp3"
    }
    // ... 3 more recommendations
  ],
  "features_extracted": 15
}
```

### Error Response
```json
{
  "error": "Prediction failed"
}
```

## Simple Logic Explanation

### 1. Feature Extraction Logic
```python
# Extract 15 audio features using librosa
features = [
    tempo_mean,           # Rhythm speed
    energy_rms_mean,      # Loudness
    spectral_centroid,    # Brightness
    zero_crossing_rate,   # Complexity
    spectral_rolloff,     # High frequencies
    mfcc_1, mfcc_2,      # Timbre
    chroma_mean,          # Harmony
    onset_strength,       # Attack
    spectral_contrast,    # Variation
    # + 6 standard deviation features
]
```

### 2. Prediction Logic
```python
# Scale features (same as training)
features_scaled = scaler.transform(features)

# Predict mood (ensemble voting)
mood = model.predict([features_scaled])[0]
probabilities = model.predict_proba([features_scaled])[0]
```

### 3. Recommendation Logic
```python
# Calculate similarity to all database songs
similarities = cosine_similarity([query_features], database)[0]

# Get top 5 most similar
top_5_indices = np.argsort(similarities)[::-1][:5]
```

## Deployment Instructions

### 1. Setup Environment
```bash
# Clone repository
git clone https://github.com/harshithkumar2004/anti-overfitting-smart-playlist-match.git
cd anti-overfitting-smart-playlist-match

# Install Python dependencies
pip install scikit-learn==1.3.0 numpy==1.24.3 librosa==0.10.1

# Install Node.js dependencies
npm install
```

### 2. File Structure
```
backend/
├── server.js
├── package.json
├── predict.py
├── anti_overfitting_mood_classifier.pkl
├── uploads/
├── audio/
└── public/
```

### 3. Start Server
```bash
npm start
# or for development
npm run dev
```

### 4. Test API
```bash
# Health check
curl http://localhost:3000/health

# Upload audio file
curl -X POST -F "audio=@test.mp3" http://localhost:3000/predict
```

## Performance Considerations

### 1. Memory Usage
- **Model**: ~239 KB
- **Feature Database**: ~50 MB
- **Total RAM**: ~100 MB

### 2. Processing Time
- **Feature Extraction**: ~0.5 seconds
- **Prediction**: ~0.01 seconds
- **Similarity Search**: ~0.1 seconds
- **Total**: ~0.6 seconds per request

### 3. Concurrency
- Python shell is synchronous
- Consider worker threads for high traffic
- Cache frequent predictions

## Troubleshooting

### Common Issues

1. **Python Path Error**
   ```javascript
   pythonPath: 'python3' // Use 'python3' instead of 'python'
   ```

2. **Model Loading Error**
   ```python
   # Ensure model file is in correct path
   MODEL_PATH = Path(__file__).parent / 'anti_overfitting_mood_classifier.pkl'
   ```

3. **Audio Processing Error**
   ```python
   # Handle unsupported formats
   try:
       features = extract_features(audio_path)
   except Exception as e:
       return None
   ```

4. **Memory Issues**
   ```javascript
   // Clean up uploaded files
   fs.unlink(audioPath).catch(console.error);
   ```

### Debug Mode
```javascript
const options = {
    mode: 'json',
    pythonPath: 'python',
    scriptPath: __dirname,
    args: [audioPath],
    // Add debugging
    stderrParser: 'stderr'
};
```

## Security Considerations

### 1. File Upload Security
```javascript
const fileFilter = (req, file, cb) => {
    if (file.mimetype.startsWith('audio/')) {
        cb(null, true);
    } else {
        cb(new Error('Only audio files allowed'), false);
    }
};

const upload = multer({ 
    storage: storage,
    fileFilter: fileFilter,
    limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});
```

### 2. Rate Limiting
```javascript
const rateLimit = require('express-rate-limit');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});

app.use('/predict', limiter);
```

## Monitoring

### 1. Request Logging
```javascript
app.post('/predict', upload.single('audio'), async (req, res) => {
    const startTime = Date.now();
    
    // ... prediction logic ...
    
    const duration = Date.now() - startTime;
    console.log(`Prediction completed in ${duration}ms`);
});
```

### 2. Health Monitoring
```javascript
app.get('/health', (req, res) => {
    const health = {
        status: 'healthy',
        model: 'anti_overfitting_mood_classifier',
        version: '1.0',
        uptime: process.uptime(),
        memory: process.memoryUsage()
    };
    res.json(health);
});
```

This integration guide provides everything needed to successfully integrate the anti-overfitting mood classifier into a Node.js/Express backend with proper error handling, security, and monitoring.
