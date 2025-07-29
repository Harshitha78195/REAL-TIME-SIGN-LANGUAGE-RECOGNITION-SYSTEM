import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os
from collections import Counter

def analyze_dataset():
    """Analyze the dataset for issues"""
    print("=== DATASET ANALYSIS ===")
    
    # Load data
    train_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')
    
    # Check class distribution
    train_counts = Counter(train_df['label'])
    test_counts = Counter(test_df['label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Number of classes: {len(train_counts)}")
    
    # Check for class imbalance
    print("\n--- Class Distribution (Training) ---")
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
              'V', 'W', 'X', 'Y']
    
    max_samples = max(train_counts.values())
    min_samples = min(train_counts.values())
    
    for i in sorted(train_counts.keys()):
        letter = labels[i] if i < len(labels) else f"Class_{i}"
        count = train_counts[i]
        percentage = count / len(train_df) * 100
        print(f"{letter}: {count:4d} samples ({percentage:5.1f}%)")
    
    print(f"\nClass imbalance ratio: {max_samples/min_samples:.2f}:1")
    
    # Check if Q is dominant
    q_class = None
    for i, letter in enumerate(labels):
        if letter == 'Q':
            q_class = i
            break
    
    if q_class is not None:
        q_samples = train_counts.get(q_class, 0)
        q_percentage = q_samples / len(train_df) * 100
        print(f"\nQ class ({q_class}): {q_samples} samples ({q_percentage:.1f}%)")
        
        if q_percentage > 8:  # More than 2x expected (100/24 = 4.17%)
            print("⚠️  Q class is over-represented! This explains the bias.")
    
    # Visualize some samples
    visualize_samples(train_df, labels)
    
    return train_counts, test_counts, labels

def visualize_samples(train_df, labels, samples_per_class=3):
    """Visualize sample images from each class"""
    print("\n=== SAMPLE VISUALIZATION ===")
    
    unique_labels = sorted(train_df['label'].unique())
    n_classes = len(unique_labels)
    
    fig, axes = plt.subplots(n_classes, samples_per_class, 
                            figsize=(samples_per_class*2, n_classes*2))
    fig.suptitle('Sample Images from Each Class', fontsize=16)
    
    for row, class_label in enumerate(unique_labels):
        class_data = train_df[train_df['label'] == class_label]
        
        for col in range(samples_per_class):
            if col < len(class_data):
                # Get image data
                img_data = class_data.iloc[col].drop('label').values
                img = img_data.reshape(28, 28)
                
                ax = axes[row, col] if n_classes > 1 else axes[col]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{labels[class_label]} (Class {class_label})')
                ax.axis('off')
            else:
                ax = axes[row, col] if n_classes > 1 else axes[col]
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_balanced_model(num_classes, class_weights=None):
    """Create a model with balanced training"""
    model = keras.Sequential([
        # Ultra-simple architecture
        layers.Conv2D(8, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((4, 4)),
        layers.Dropout(0.3),
        
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),  # Instead of flatten + dense
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Use balanced class weights if provided
    if class_weights:
        print("Using balanced class weights to handle imbalance")
    
    return model

def train_balanced_model():
    """Train with balanced approach"""
    print("=== BALANCED MODEL TRAINING ===")
    
    # Analyze dataset first
    train_counts, test_counts, labels = analyze_dataset()
    
    # Load data
    train_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')
    
    X_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Balance the dataset by undersampling
    print("\n=== BALANCING DATASET ===")
    min_samples = min(train_counts.values())
    print(f"Undersampling all classes to {min_samples} samples each")
    
    balanced_X = []
    balanced_y = []
    
    for class_label in sorted(train_counts.keys()):
        class_mask = y_train_full == class_label
        class_X = X_train_full[class_mask]
        class_y = y_train_full[class_mask]
        
        # Randomly sample min_samples from this class
        indices = np.random.choice(len(class_X), min_samples, replace=False)
        balanced_X.append(class_X[indices])
        balanced_y.append(class_y[indices])
    
    X_balanced = np.vstack(balanced_X)
    y_balanced = np.hstack(balanced_y)
    
    print(f"Balanced dataset: {len(X_balanced)} samples")
    print(f"Samples per class: {min_samples}")
    
    # Create train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, 
        stratify=y_balanced
    )
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    num_classes = len(np.unique(y_balanced))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # Create model
    model = create_balanced_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n=== MODEL ARCHITECTURE ===")
    model.summary()
    
    # Train
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    print("\n=== TRAINING ===")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n=== RESULTS ===")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Test individual class predictions
    test_predictions_per_class(model, X_test, y_test, labels)
    
    # Save model
    model.save('balanced_sign_model.keras')
    print("Model saved as 'balanced_sign_model.keras'")
    
    return model, history

def test_predictions_per_class(model, X_test, y_test, labels):
    """Test how well the model predicts each class"""
    print("\n=== PER-CLASS PERFORMANCE ===")
    
    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(y_test, axis=1)
    
    # Calculate accuracy per class
    class_accuracies = {}
    for class_idx in range(len(labels)):
        class_mask = true_classes == class_idx
        if np.sum(class_mask) > 0:  # If this class exists in test set
            class_preds = predicted_classes[class_mask]
            class_acc = np.mean(class_preds == class_idx)
            class_accuracies[class_idx] = class_acc
            
            letter = labels[class_idx] if class_idx < len(labels) else f"Class_{class_idx}"
            print(f"{letter}: {class_acc:.3f} ({np.sum(class_mask)} samples)")
    
    # Check if model always predicts the same class
    unique_predictions = len(np.unique(predicted_classes))
    print(f"\nModel predicts {unique_predictions} different classes out of {len(labels)}")
    
    if unique_predictions < 5:
        print("⚠️  WARNING: Model is only predicting a few classes!")
        most_common = Counter(predicted_classes).most_common(3)
        print("Most predicted classes:")
        for class_idx, count in most_common:
            letter = labels[class_idx] if class_idx < len(labels) else f"Class_{class_idx}"
            percentage = count / len(predicted_classes) * 100
            print(f"  {letter}: {count} times ({percentage:.1f}%)")

if __name__ == "__main__":
    print("=== DIAGNOSTIC SIGN LANGUAGE TRAINING ===")
    print("This will analyze your dataset and train a balanced model")
    print()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    try:
        model, history = train_balanced_model()
        print("\n" + "="*50)
        print("DIAGNOSTIC TRAINING COMPLETED!")
        print("="*50)
        print("Check the output above to see:")
        print("1. Class distribution analysis")
        print("2. Per-class performance")
        print("3. Whether the model is biased to specific classes")
        print("\nFiles created:")
        print("- balanced_sign_model.keras")
        print("- dataset_samples.png")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("Make sure sign_mnist_train.csv and sign_mnist_test.csv exist in the current directory")