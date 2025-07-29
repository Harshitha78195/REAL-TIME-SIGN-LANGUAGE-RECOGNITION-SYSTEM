#!/usr/bin/env python3
"""
Main script to run the Sign Language Recognition System
Provides a menu interface to access all functionalities
"""

import os
import sys
import subprocess

def check_files():
    """Check if required files exist"""
    required_files = [
        'sign_mnist_train.csv',
        'sign_mnist_test.csv',
        'requirements.txt',
        'train_model.py',
        'hand_detection.py',
        'real_time_recognition.py',
        'test_model.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all files are in the current directory.")
        return False
    
    print("✅ All required files found!")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import mediapipe
        import tensorflow
        import numpy
        import pandas
        import matplotlib
        import sklearn
        import PIL
        print("✅ All dependencies are installed!")
        return True
    except ImportError as e:
        print("❌ Missing dependencies!")
        print(f"Error: {e}")
        print("\nPlease install dependencies using:")
        print("pip install -r requirements.txt")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies!")
        print("Please install manually using: pip install -r requirements.txt")
        return False

def train_model():
    """Train the sign language model"""
    print("\n" + "="*50)
    print("TRAINING SIGN LANGUAGE MODEL")
    print("="*50)
    print("This will train a CNN model on the Sign Language MNIST dataset.")
    print("Training may take 10-30 minutes depending on your hardware.")
    print("\nPress Enter to continue or 'q' to cancel...")
    
    choice = input().strip().lower()
    if choice == 'q':
        return
    
    try:
        subprocess.run([sys.executable, "train_model.py"], check=True)
        print("\n✅ Model training completed!")
        print("Generated files:")
        print("   - sign_language_model.h5")
        print("   - best_sign_model.h5") 
        print("   - label_mapping.json")
        print("   - training_history.png")
    except subprocess.CalledProcessError:
        print("❌ Model training failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

def test_model():
    """Test the trained model"""
    print("\n" + "="*50)
    print("TESTING SIGN LANGUAGE MODEL")
    print("="*50)
    
    if not os.path.exists('sign_language_model.h5'):
        print("❌ Model not found! Please train the model first.")
        return
    
    try:
        subprocess.run([sys.executable, "test_model.py"], check=True)
        print("\n✅ Model testing completed!")
    except subprocess.CalledProcessError:
        print("❌ Model testing failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Testing interrupted by user")

def run_real_time():
    """Run real-time recognition"""
    print("\n" + "="*50)
    print("REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("="*50)
    
    if not os.path.exists('sign_language_model.h5'):
        print("❌ Model not found! Please train the model first.")
        return
    
    print("Starting real-time recognition...")
    print("Make sure your webcam is connected and working.")
    print("\nControls:")
    print("   SPACE: Add word")
    print("   BACKSPACE: Delete last letter")
    print("   C: Clear current word")
    print("   R: Reset all")
    print("   S: Save sentence")
    print("   Q: Quit")
    
    input("\nPress Enter to start...")
    
    try:
        subprocess.run([sys.executable, "real_time_recognition.py"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Real-time recognition failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Recognition stopped by user")

def quick_webcam_test():
    """Quick webcam test"""
    print("\n" + "="*50)
    print("QUICK WEBCAM TEST")
    print("="*50)
    
    if not os.path.exists('sign_language_model.h5'):
        print("❌ Model not found! Please train the model first.")
        return
    
    print("Testing webcam with basic sign recognition...")
    print("Press 'q' in the video window to quit.")
    
    input("Press Enter to start webcam test...")
    
    try:
        import subprocess
        subprocess.run([sys.executable, "-c", """
from hand_detection import SignLanguagePredictor
import cv2

predictor = SignLanguagePredictor()
if predictor.model is None:
    print("Model not found!")
    exit()

cap = cv2.VideoCapture(0)
print("Webcam test - Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    results = predictor.hand_detector.detect_hands(frame)
    prediction_info = predictor.predict_sign(frame)
    annotated_frame = predictor.draw_prediction(frame, results, prediction_info)
    
    cv2.putText(annotated_frame, "Webcam Test - Press 'q' to quit", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam Test', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam test completed!")
        """], check=True)
    except Exception as e:
        print(f"❌ Webcam test failed: {e}")

def show_system_info():
    """Show system information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check file sizes
    print("\nDataset files:")
    for file in ['sign_mnist_train.csv', 'sign_mnist_test.csv']:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024 / 1024  # MB
            print(f"   {file}: {size:.1f} MB")
        else:
            print(f"   {file}: NOT FOUND")
    
    # Check model file
    print("\nModel files:")
    for file in ['sign_language_model.h5', 'label_mapping.json']:
        if os.path.exists(file):
            if file.endswith('.h5'):
                size = os.path.getsize(file) / 1024 / 1024  # MB
                print(f"   {file}: {size:.1f} MB")
            else:
                print(f"   {file}: EXISTS")
        else:
            print(f"   {file}: NOT FOUND")
    
    # Check dependencies
    print("\nDependency versions:")
    packages = ['opencv-python', 'mediapipe', 'tensorflow', 'numpy', 'pandas']
    for package in packages:
        try:
            if package == 'opencv-python':
                import cv2
                print(f"   {package}: {cv2.__version__}")
            elif package == 'mediapipe':
                import mediapipe as mp
                print(f"   {package}: {mp.__version__}")
            elif package == 'tensorflow':
                import tensorflow as tf
                print(f"   {package}: {tf.__version__}")
            elif package == 'numpy':
                import numpy as np
                print(f"   {package}: {np.__version__}")
            elif package == 'pandas':
                import pandas as pd
                print(f"   {package}: {pd.__version__}")
        except ImportError:
            print(f"   {package}: NOT INSTALLED")

def show_menu():
    """Show main menu"""
    print("\n" + "="*50)
    print("SIGN LANGUAGE RECOGNITION SYSTEM")
    print("="*50)
    print("1. Install Dependencies")
    print("2. Train Model")
    print("3. Test Model")
    print("4. Run Real-time Recognition")
    print("5. Quick Webcam Test")
    print("6. System Information")
    print("7. Exit")
    print("="*50)

def main():
    """Main function"""
    print("🤟 Welcome to Sign Language Recognition System!")
    
    # Initial checks
    if not check_files():
        print("\nPlease ensure all required files are present and try again.")
        return
    
    while True:
        show_menu()
        choice = input("Enter your choice (1-7): ").strip()
        
        if choice == '1':
            install_dependencies()
        elif choice == '2':
            if check_dependencies():
                train_model()
            else:
                print("Please install dependencies first (option 1)")
        elif choice == '3':
            if check_dependencies():
                test_model()
            else:
                print("Please install dependencies first (option 1)")
        elif choice == '4':
            if check_dependencies():
                run_real_time()
            else:
                print("Please install dependencies first (option 1)")
        elif choice == '5':
            if check_dependencies():
                quick_webcam_test()
            else:
                print("Please install dependencies first (option 1)")
        elif choice == '6':
            show_system_info()
        elif choice == '7':
            print("\n👋 Thank you for using Sign Language Recognition System!")
            break
        else:
            print("❌ Invalid choice! Please enter 1-7.")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again.")