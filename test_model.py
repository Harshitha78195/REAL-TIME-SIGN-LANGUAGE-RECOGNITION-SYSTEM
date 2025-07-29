import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import json
import cv2
from hand_detection import SignLanguagePredictor

def load_test_data():
    """Load test data"""
    test_df = pd.read_csv('sign_mnist_test.csv')
    
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Reshape to 28x28 images
    X_test = X_test.reshape(-1, 28, 28, 1)
    X_test = X_test.astype('float32') / 255.0
    
    return X_test, y_test

def evaluate_model():
    """Evaluate the trained model"""
    # Load model
    try:
        model = tf.keras.models.load_model('sign_language_model.h5')
        print("Model loaded successfully!")
    except:
        print("Error: Could not load model. Please train the model first.")
        return
    
    # Load label mapping
    try:
        with open('label_mapping.json', 'r') as f:
            label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
        print("Label mapping loaded!")
    except:
        print("Error: Could not load label mapping.")
        return
    
    # Load test data
    X_test, y_test = load_test_data()
    print(f"Test data shape: {X_test.shape}")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Calculate accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Create classification report
    class_names = [label_map[i] for i in range(len(label_map))]
    print("\nClassification Report:")
    print(classification_report(y_test, predicted_classes, target_names=class_names))
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, predicted_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Show some predictions
    show_sample_predictions(X_test, y_test, predicted_classes, predictions, label_map)
    
    return accuracy, cm

def show_sample_predictions(X_test, y_test, predicted_classes, predictions, label_map, n_samples=16):
    """Show sample predictions"""
    # Select random samples
    indices = np.random.choice(len(X_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Get image and predictions
        image = X_test[idx].reshape(28, 28)
        true_label = label_map[y_test[idx]]
        pred_label = label_map[predicted_classes[idx]]
        confidence = predictions[idx][predicted_classes[idx]]
        
        # Plot image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}', 
                          fontsize=10)
        axes[i].axis('off')
        
        # Color based on correctness
        if true_label == pred_label:
            axes[i].patch.set_edgecolor('green')
            axes[i].patch.set_linewidth(2)
        else:
            axes[i].patch.set_edgecolor('red')
            axes[i].patch.set_linewidth(2)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def test_with_webcam():
    """Test model with webcam"""
    print("Testing with webcam...")
    
    try:
        predictor = SignLanguagePredictor()
        
        if predictor.model is None:
            print("Model not found. Please train the model first.")
            return
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Webcam test started! Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip frame
            frame = cv2.flip(frame, 1)
            
            # Predict
            results = predictor.hand_detector.detect_hands(frame)
            prediction_info = predictor.predict_sign(frame)
            
            # Draw results
            annotated_frame = predictor.draw_prediction(frame, results, prediction_info)
            
            # Add test info
            cv2.putText(annotated_frame, "Model Test Mode", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Model Test', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error during webcam test: {e}")

def analyze_model_performance():
    """Analyze model performance in detail"""
    try:
        model = tf.keras.models.load_model('sign_language_model.h5')
    except:
        print("Error: Could not load model.")
        return
    
    # Load test data
    X_test, y_test = load_test_data()
    
    # Make predictions
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Load label mapping
    with open('label_mapping.json', 'r') as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}
    
    # Per-class accuracy
    class_accuracies = {}
    for class_id in range(len(label_map)):
        class_mask = y_test == class_id
        if np.sum(class_mask) > 0:
            class_acc = np.mean(predicted_classes[class_mask] == y_test[class_mask])
            class_accuracies[label_map[class_id]] = class_acc
    
    # Sort by accuracy
    sorted_accuracies = sorted(class_accuracies.items(), key=lambda x: x[1])
    
    print("\nPer-class Accuracy (sorted from lowest to highest):")
    print("-" * 50)
    for letter, acc in sorted_accuracies:
        print(f"{letter}: {acc:.4f}")
    
    # Plot per-class accuracy
    letters, accuracies = zip(*sorted_accuracies)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(letters, accuracies, color=['red' if acc < 0.8 else 'orange' if acc < 0.9 else 'green' for acc in accuracies])
    plt.title('Per-class Accuracy')
    plt.xlabel('Letters')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # Add accuracy labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Find most confused pairs
    cm = confusion_matrix(y_test, predicted_classes)
    
    confusion_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                confusion_pairs.append((label_map[i], label_map[j], cm[i][j]))
    
    # Sort by confusion count
    confusion_pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop 10 Most Confused Letter Pairs:")
    print("-" * 50)
    for i, (true_letter, pred_letter, count) in enumerate(confusion_pairs[:10]):
        print(f"{i+1}. {true_letter} -> {pred_letter}: {count} times")

def main():
    """Main testing function"""
    print("Sign Language Model Testing")
    print("=" * 40)
    
    print("\n1. Evaluating model performance...")
    accuracy, cm = evaluate_model()
    
    print("\n2. Analyzing detailed performance...")
    analyze_model_performance()
    
    print(f"\n3. Overall Test Accuracy: {accuracy:.4f}")
    
    # Ask if user wants to test with webcam
    choice = input("\nDo you want to test with webcam? (y/n): ").lower()
    if choice == 'y':
        test_with_webcam()
    
    print("\nTesting completed!")
    print("Generated files:")
    print("- confusion_matrix.png")
    print("- sample_predictions.png") 
    print("- per_class_accuracy.png")

if __name__ == "__main__":
    main()