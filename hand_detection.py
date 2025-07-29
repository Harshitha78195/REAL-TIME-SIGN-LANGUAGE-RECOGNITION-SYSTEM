import cv2
import numpy as np
import tensorflow as tf
import json
import os
import mediapipe as mp
from PIL import Image, ImageDraw

class SimpleCorrectHandDetector:
    def __init__(self):
        """Simple hand detector focused on matching training data exactly"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
    
    def detect_hand_landmarks(self, frame):
        """Get hand landmarks"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None
    
    def create_training_data_style_image(self, frame, hand_landmarks):
        """Create image that matches the training data style exactly"""
        if hand_landmarks is None:
            return None, None
        
        h, w, _ = frame.shape
        
        # Get hand landmark coordinates
        landmark_points = []
        for lm in hand_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_points.append((x, y))
        
        # Find bounding box of landmarks
        x_coords = [p[0] for p in landmark_points]
        y_coords = [p[1] for p in landmark_points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add padding
        padding = 40
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)  
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Extract region
        roi = frame[y_min:y_max, x_min:x_max]
        
        if roi.size == 0:
            return None, None
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray_roi = roi.copy()
        
        # Make it square by padding with black
        roi_h, roi_w = gray_roi.shape
        max_dim = max(roi_h, roi_w)
        
        # Create square black image
        square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
        
        # Center the ROI in the square
        start_y = (max_dim - roi_h) // 2
        start_x = (max_dim - roi_w) // 2
        square_img[start_y:start_y+roi_h, start_x:start_x+roi_w] = gray_roi
        
        # Resize to 64x64 first for better quality, then to 28x28
        resized_64 = cv2.resize(square_img, (64, 64), interpolation=cv2.INTER_AREA)
        resized_28 = cv2.resize(resized_64, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Apply edge-preserving filtering
        filtered = cv2.bilateralFilter(resized_28, 9, 75, 75)
        
        # Create binary mask using multiple thresholding methods
        # Method 1: Otsu thresholding
        _, otsu_thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive thresholding  
        adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Simple threshold at a reasonable level
        _, simple_thresh = cv2.threshold(filtered, 60, 255, cv2.THRESH_BINARY)
        
        # Combine thresholding methods (take the one with most reasonable hand size)
        contours_otsu, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_adaptive, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_simple, _ = cv2.findContours(simple_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Choose the thresholding that gives the most reasonable hand area
        def get_largest_contour_area(contours):
            if not contours:
                return 0
            return cv2.contourArea(max(contours, key=cv2.contourArea))
        
        otsu_area = get_largest_contour_area(contours_otsu)
        adaptive_area = get_largest_contour_area(contours_adaptive)
        simple_area = get_largest_contour_area(contours_simple)
        
        # Choose the method that gives area between 100-500 pixels (reasonable for 28x28)
        areas = [(otsu_area, otsu_thresh), (adaptive_area, adaptive_thresh), (simple_area, simple_thresh)]
        
        # Filter areas that are reasonable (between 50 and 600 pixels for 28x28 image)
        reasonable_areas = [(area, thresh) for area, thresh in areas if 50 <= area <= 600]
        
        if reasonable_areas:
            # Choose the one closest to 200 pixels (good size for 28x28)
            _, binary = min(reasonable_areas, key=lambda x: abs(x[0] - 200))
        else:
            # Fallback to adaptive if no reasonable area found
            binary = adaptive_thresh
        
        # Clean up the binary image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Ensure hand is white on black background (like training data)
        # Count pixels in center region
        center_region = binary[10:18, 10:18]
        center_white = np.sum(center_region == 255)
        center_black = np.sum(center_region == 0)
        
        # If more black in center, the hand is probably black - invert
        if center_black > center_white:
            binary = cv2.bitwise_not(binary)
        
        # Final cleanup
        binary = cv2.medianBlur(binary, 3)
        
        # Normalize for model
        normalized = binary.astype('float32') / 255.0
        model_input = normalized.reshape(1, 28, 28, 1)
        
        bbox = (x_min, y_min, x_max, y_max)
        
        return model_input, binary, bbox

class SimpleSignLanguagePredictor:
    def __init__(self, model_path='conservative_sign_model.keras', label_map_path='label_mapping.json'):
        self.hand_detector = SimpleCorrectHandDetector()
        
        # Load model
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Model file {model_path} not found.")
            self.model = None
        
        # Load label mapping
        if os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.label_map = json.load(f)
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            print(f"Label mapping loaded: {list(self.label_map.values())}")
        else:
            print(f"Label mapping file {label_map_path} not found.")
            self.label_map = None
        
        # Prediction history for stability
        self.prediction_history = []
        self.history_size = 5
    
    def predict_sign(self, frame):
        """Predict sign from frame"""
        if self.model is None or self.label_map is None:
            return None, 0.0, None
        
        # Detect hand landmarks
        hand_landmarks = self.hand_detector.detect_hand_landmarks(frame)
        
        if hand_landmarks is None:
            return None, 0.0, None
        
        # Create training-data style image
        result = self.hand_detector.create_training_data_style_image(frame, hand_landmarks)
        
        if result[0] is None:
            return None, 0.0, None
        
        model_input, processed_image, bbox = result
        
        try:
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            # Get top 3 predictions for debugging
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_predictions = [(self.label_map.get(i, f"Class_{i}"), predictions[0][i]) 
                               for i in top_3_indices]
            
            predicted_letter = self.label_map.get(predicted_class, "Unknown")
            
            # Add to history
            self.prediction_history.append((predicted_letter, confidence))
            if len(self.prediction_history) > self.history_size:
                self.prediction_history.pop(0)
            
            # Get stable prediction
            stable_letter, stable_confidence = self.get_stable_prediction()
            
            return stable_letter, stable_confidence, (bbox, processed_image, hand_landmarks, top_3_predictions)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, 0.0, None
    
    def get_stable_prediction(self, min_confidence=0.4):
        """Get stable prediction from history"""
        if len(self.prediction_history) < 3:
            return None, 0.0
        
        # Get recent predictions with decent confidence
        recent_good = [(letter, conf) for letter, conf in self.prediction_history[-3:] 
                      if conf >= min_confidence]
        
        if not recent_good:
            return None, 0.0
        
        # Find most common letter
        from collections import Counter
        letter_counts = Counter([letter for letter, _ in recent_good])
        
        if letter_counts:
            most_common_letter = letter_counts.most_common(1)[0][0]
            # Get average confidence for this letter
            confidences = [conf for letter, conf in recent_good if letter == most_common_letter]
            avg_confidence = sum(confidences) / len(confidences)
            
            return most_common_letter, avg_confidence
        
        return None, 0.0
    
    def draw_prediction(self, frame, prediction_info):
        """Draw prediction results"""
        annotated_frame = frame.copy()
        
        if prediction_info[0] is not None:
            letter, confidence, extra_info = prediction_info
            
            if extra_info and len(extra_info) >= 4:
                bbox, processed_image, hand_landmarks, top_3_preds = extra_info
                x1, y1, x2, y2 = bbox
                
                # Draw hand landmarks
                if hand_landmarks:
                    self.hand_detector.mp_draw.draw_landmarks(
                        annotated_frame, hand_landmarks,
                        self.hand_detector.mp_hands.HAND_CONNECTIONS
                    )
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw main prediction
                if letter and confidence > 0.3:
                    text = f"{letter}: {confidence:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1 - 40), (x1 + 200, y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, text, (x1 + 5, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                
                # Show processed image (larger)
                if processed_image is not None:
                    display_size = 140
                    try:
                        # Show the processed image with better scaling
                        processed_display = cv2.resize(processed_image, (display_size, display_size), 
                                                     interpolation=cv2.INTER_NEAREST)
                        processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
                        
                        h, w = annotated_frame.shape[:2]
                        annotated_frame[10:10+display_size, w-display_size-10:w-10] = processed_display
                        
                        cv2.rectangle(annotated_frame, (w-display_size-10, 10), 
                                    (w-10, 10+display_size), (255, 255, 255), 2)
                        cv2.putText(annotated_frame, "Model Input", (w-display_size-5, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    except Exception as e:
                        print(f"Display error: {e}")
                
                # Show top 3 predictions
                if top_3_preds:
                    y_offset = 200
                    cv2.putText(annotated_frame, "Top 3:", (w-200, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    for i, (pred_letter, pred_conf) in enumerate(top_3_preds):
                        y_pos = y_offset + 25 + (i * 20)
                        text = f"{i+1}. {pred_letter}: {pred_conf:.3f}"
                        cv2.putText(annotated_frame, text, (w-200, y_pos), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame

def test_simple_predictor():
    """Test the simple predictor"""
    predictor = SimpleSignLanguagePredictor()
    
    if predictor.model is None:
        print("Model not found!")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Simple Sign Language Recognition")
    print("Available letters:", list(predictor.label_map.values()))
    print("Hold your hand steady and make clear signs")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        # Predict
        prediction_info = predictor.predict_sign(frame)
        
        # Draw results
        annotated_frame = predictor.draw_prediction(frame, prediction_info)
        
        # Add title
        cv2.putText(annotated_frame, "Simple Sign Language Recognition", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_frame, "Press 'q' to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow('Simple Sign Language Recognition', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_simple_predictor()