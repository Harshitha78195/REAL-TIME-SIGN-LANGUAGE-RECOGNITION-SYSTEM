import cv2
import numpy as np
from hand_detection import SignLanguagePredictor
import time
import threading
import queue

class RealTimeSignRecognition:
    def __init__(self):
        """Initialize real-time sign language recognition"""
        self.predictor = SignLanguagePredictor()
        
        if self.predictor.model is None:
            raise ValueError("Model not found. Please train the model first.")
        
        # Word building
        self.current_word = ""
        self.last_letter = ""
        self.letter_start_time = 0
        self.letter_hold_time = 1.5  # seconds to hold letter before adding
        self.word_list = []
        
        # Threading for smooth video
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.processing = True
        
    def process_frames(self):
        """Process frames in separate thread"""
        while self.processing:
            try:
                frame = self.frame_queue.get(timeout=1)
                if frame is None:
                    break
                
                # Detect hands and predict
                results = self.predictor.hand_detector.detect_hands(frame)
                prediction_info = self.predictor.predict_sign(frame)
                
                self.result_queue.put((frame, results, prediction_info))
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def update_word_building(self, letter, confidence, min_confidence=0.8):
        """Update word building based on detected letter"""
        current_time = time.time()
        
        if letter and confidence >= min_confidence:
            if letter != self.last_letter:
                # New letter detected
                self.last_letter = letter
                self.letter_start_time = current_time
            elif current_time - self.letter_start_time >= self.letter_hold_time:
                # Letter held long enough, add to word
                if letter not in self.current_word or self.current_word[-1] != letter:
                    self.current_word += letter
                    self.letter_start_time = current_time  # Reset timer
        else:
            # No confident detection
            self.last_letter = ""
            self.letter_start_time = 0
    
    def draw_interface(self, frame, results, prediction_info):
        """Draw the complete interface"""
        # Draw prediction and hand landmarks
        annotated_frame = self.predictor.draw_prediction(frame, results, prediction_info)
        
        h, w = annotated_frame.shape[:2]
        
        # Draw word building section
        word_section_height = 150
        cv2.rectangle(annotated_frame, (0, h - word_section_height), (w, h), (50, 50, 50), -1)
        
        # Current word
        cv2.putText(annotated_frame, f"Current Word: {self.current_word}", 
                   (10, h - word_section_height + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Recent words
        recent_words = " ".join(self.word_list[-5:])  # Show last 5 words
        cv2.putText(annotated_frame, f"Recent: {recent_words}", 
                   (10, h - word_section_height + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        # Letter hold progress
        if self.last_letter and prediction_info[0]:
            current_time = time.time()
            if self.letter_start_time > 0:
                progress = min((current_time - self.letter_start_time) / self.letter_hold_time, 1.0)
                bar_length = 200
                bar_height = 10
                bar_x = 10
                bar_y = h - word_section_height + 110
                
                # Progress bar background
                cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                            (bar_x + bar_length, bar_y + bar_height), (100, 100, 100), -1)
                
                # Progress bar fill
                fill_length = int(bar_length * progress)
                cv2.rectangle(annotated_frame, (bar_x, bar_y), 
                            (bar_x + fill_length, bar_y + bar_height), (0, 255, 0), -1)
                
                cv2.putText(annotated_frame, f"Hold '{self.last_letter}' to add", 
                           (bar_x + bar_length + 10, bar_y + bar_height), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        instructions = [
            "Instructions:",
            "- Hold hand sign for 1.5 seconds to add letter",
            "- Press SPACE to add word",
            "- Press BACKSPACE to delete last letter",
            "- Press 'c' to clear current word",
            "- Press 'r' to reset all",
            "- Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            y_pos = 30 + i * 25
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            thickness = 2 if i == 0 else 1
            cv2.putText(annotated_frame, instruction, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
        
        return annotated_frame
    
    def run(self):
        """Run the real-time recognition system"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Start processing thread
        processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        processing_thread.start()
        
        print("Real-time Sign Language Recognition Started!")
        print("Available letters:", list(self.predictor.label_map.values()))
        print("Follow the on-screen instructions")
        
        last_frame = None
        last_results = None
        last_prediction = (None, 0.0, None)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Add frame to processing queue
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                
                # Get results if available
                try:
                    current_frame, results, prediction_info = self.result_queue.get_nowait()
                    last_frame = current_frame
                    last_results = results
                    last_prediction = prediction_info
                except queue.Empty:
                    pass
                
                # Update word building
                if last_prediction[0]:
                    letter, confidence, _ = last_prediction
                    self.update_word_building(letter, confidence)
                
                # Draw interface
                if last_frame is not None:
                    display_frame = self.draw_interface(frame, last_results, last_prediction)
                else:
                    display_frame = frame
                
                cv2.imshow('Real-time Sign Language Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Space - add word
                    if self.current_word:
                        self.word_list.append(self.current_word)
                        print(f"Added word: '{self.current_word}'")
                        self.current_word = ""
                elif key == 8:  # Backspace - delete last letter
                    if self.current_word:
                        self.current_word = self.current_word[:-1]
                elif key == ord('c'):  # Clear current word
                    self.current_word = ""
                    print("Cleared current word")
                elif key == ord('r'):  # Reset all
                    self.current_word = ""
                    self.word_list = []
                    print("Reset all words")
                elif key == ord('s'):  # Save sentence
                    if self.word_list or self.current_word:
                        sentence = " ".join(self.word_list)
                        if self.current_word:
                            sentence += " " + self.current_word
                        
                        with open('recognized_text.txt', 'a') as f:
                            f.write(sentence.strip() + "\n")
                        
                        print(f"Saved: '{sentence.strip()}'")
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        finally:
            # Cleanup
            self.processing = False
            self.frame_queue.put(None)  # Signal processing thread to stop
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final results
            if self.word_list or self.current_word:
                print("\nFinal sentence:")
                sentence = " ".join(self.word_list)
                if self.current_word:
                    sentence += " " + self.current_word
                print(f"'{sentence.strip()}'")

def main():
    """Main function"""
    try:
        recognizer = RealTimeSignRecognition()
        recognizer.run()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please run 'python train_model.py' first to train the model.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()