# REAL-TIME-SIGN-LANGUAGE-RECOGNITION-SYSTEM

A complete real-time sign language recognition system using the Sign Language MNIST dataset, MediaPipe for hand detection, and TensorFlow for classification.

## Features

- **Real-time hand detection** using MediaPipe
- **CNN-based sign classification** trained on Sign Language MNIST dataset
- **Word building functionality** with letter hold detection
- **Smooth prediction** with confidence-based filtering
- **Interactive interface** with live feedback
- **Model evaluation tools** with detailed performance metrics

## Dataset

The system uses the Sign Language MNIST dataset which contains:
- 24 classes of American Sign Language letters (A-Y, excluding J and Z)
- 27,455 training samples and 7,172 test samples
- 28x28 grayscale images
- Letters J and Z are excluded as they require motion

## File Structure

```
SIGN_LANGUAGE/
├── sign_mnist_train.csv          # Training dataset
├── sign_mnist_test.csv           # Test dataset
├── amer_sign2.png               # Reference images
├── amer_sign3.png
├── american_sign_language.png
├── requirements.txt             # Python dependencies
├── train_model.py              # Model training script
├── hand_detection.py           # Hand detection and prediction
├── real_time_recognition.py    # Real-time recognition system
├── test_model.py              # Model evaluation tools
└── README.md                  # This file
```

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure you have the dataset files:**
   - `sign_mnist_train.csv`
   - `sign_mnist_test.csv`

## Usage

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a CNN model with data augmentation
- Save the trained model as `conservative_sign_model.keras`
- Generate training plots and label mappings
- Create `label_mapping.json` for letter mappings

**Expected output files:**
- `conservative_sign_model.keras` - Main trained model
- `best_sign_model.h5` - Best model during training
- `label_mapping.json` - Letter to class mapping
- `training_history.png` - Training/validation curves

### Step 2: Test the Model

```bash
python test_model.py
```

This will:
- Evaluate model performance on test data
- Generate confusion matrix and accuracy metrics
- Show sample predictions
- Optionally test with webcam

**Expected output files:**
- `confusion_matrix.png` - Visual confusion matrix
- `sample_predictions.png` - Sample prediction results
- `per_class_accuracy.png` - Per-letter accuracy chart

### Step 3: Run Real-Time Recognition

```bash
python real_time_recognition.py
```

This will start the real-time recognition system with:
- Live webcam feed
- Hand detection and sign prediction
- Word building functionality
- Interactive controls

## Real-Time System Controls

| Key | Action |
|-----|--------|
| **SPACE** | Add current word to sentence |
| **BACKSPACE** | Delete last letter from current word |
| **C** | Clear current word |
| **R** | Reset all words |
| **S** | Save sentence to `recognized_text.txt` |
| **Q** | Quit application |

## How It Works

### 1. Hand Detection
- Uses MediaPipe to detect hand landmarks
- Extracts hand region with bounding box
- Converts to grayscale and resizes to 28x28

### 2. Sign Classification
- CNN model processes the hand region
- Outputs probability for each of 24 letters
- Uses confidence thresholding and smoothing

### 3. Word Building
- Requires holding a sign for 1.5 seconds
- Prevents duplicate letters in sequence
- Builds words letter by letter

### 4. Real-Time Interface
- Shows live prediction with confidence
- Displays current word and recent words
- Progress bar for letter hold timing
- Processed hand image in corner

## Model Architecture

```
Conv2D(32) -> BatchNorm -> Conv2D(32) -> MaxPool -> Dropout(0.25)
Conv2D(64) -> BatchNorm -> Conv2D(64) -> MaxPool -> Dropout(0.25)
Conv2D(128) -> BatchNorm -> Dropout(0.25)
Flatten -> Dense(512) -> BatchNorm -> Dropout(0.5)
Dense(256) -> Dropout(0.5) -> Dense(24, softmax)
```

## Performance Optimization

- **Prediction Smoothing**: Uses history of last 10 predictions
- **Confidence Filtering**: Only accepts predictions above threshold
- **Threading**: Separate thread for frame processing
- **Queue Management**: Prevents frame buffer overflow

## Troubleshooting

### Common Issues

1. **"Model not found" error:**
   - Run `python train_model.py` first
   - Check if `sign_language_model.h5` exists

2. **Webcam not working:**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)
   - Ensure no other applications are using the camera

3. **Poor recognition accuracy:**
   - Ensure good lighting conditions
   - Hold hand clearly against plain background
   - Keep hand steady for 1.5 seconds
   - Check if letter is supported (no J or Z)

4. **Installation issues:**
   - Update pip: `pip install --upgrade pip`
   - Use virtual environment
   - Check Python version (3.7+ recommended)

### Performance Tips

- **Lighting**: Use good, even lighting
- **Background**: Plain, contrasting background works best
- **Distance**: Keep hand 1-2 feet from camera
- **Stability**: Hold signs steady and clear
- **Patience**: Wait for confidence to build up

## Available Letters

The system recognizes these 24 letters:
```
A B C D E F G H I K L M N O P Q R S T U V W X Y
```

**Note**: Letters J and Z are not included as they require motion gestures.

## Output Files

- `recognized_text.txt` - Saved sentences (when pressing 'S')
- `training_history.png` - Training metrics
- `confusion_matrix.png` - Model evaluation
- `sample_predictions.png` - Prediction examples
- `per_class_accuracy.png` - Per-letter accuracy

## Technical Requirements

- **Python 3.7+**
- **Webcam** for real-time recognition
- **4GB+ RAM** for model training
- **GPU optional** but recommended for faster training

## License

This project is for educational purposes. The Sign Language MNIST dataset has its own licensing terms.

## References

- Sign Language MNIST Dataset
- MediaPipe Hand Detection
- TensorFlow/Keras Deep Learning
- OpenCV Computer Vision

## Contributing

Feel free to improve the system by:
- Adding more letters or words
- Improving the model architecture
- Enhancing the user interface
- Adding text-to-speech functionality
- Supporting different sign languages
