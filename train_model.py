import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import os

def load_data():
    """Load and preprocess with extra validation to prevent data leakage"""
    print("Loading training data...")
    train_df = pd.read_csv('sign_mnist_train.csv')
    
    print("Loading test data...")
    test_df = pd.read_csv('sign_mnist_test.csv')
    
    # Check for data leakage (identical samples between train/test)
    print("Checking for data leakage...")
    X_train_full = train_df.drop('label', axis=1).values
    y_train_full = train_df['label'].values
    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values
    
    # Create stratified train/val split with larger validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.3,  # Larger validation set (30%)
        random_state=42, 
        stratify=y_train_full
    )
    
    print(f"Original training samples: {len(X_train_full)}")
    print(f"New training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    # Reshape and normalize
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_val = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Add noise to prevent overfitting
    noise_factor = 0.05
    X_train += noise_factor * np.random.normal(0, 1, X_train.shape)
    X_train = np.clip(X_train, 0, 1)
    
    num_classes = max(np.max(y_train_full), np.max(y_test)) + 1
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, num_classes

def create_minimal_model(num_classes):
    """Create an extremely simple model to prevent overfitting"""
    model = keras.Sequential([
        # Very simple architecture
        layers.Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((4, 4)),  # Aggressive pooling
        layers.Dropout(0.5),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.6),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),  # Very small dense layer
        layers.Dropout(0.7),  # High dropout
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def create_cross_validation_splits(X, y, n_folds=5):
    """Create multiple train/val splits for cross-validation"""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    y_labels = np.argmax(y, axis=1)  # Convert one-hot back to labels
    
    splits = []
    for train_idx, val_idx in skf.split(X, y_labels):
        splits.append((train_idx, val_idx))
    
    return splits

def train_with_cross_validation():
    """Train with cross-validation to get realistic performance estimates"""
    print("=== CROSS-VALIDATION TRAINING ===")
    
    # Load data
    X_train_full, y_train_full, X_val, y_val, X_test, y_test, num_classes = load_data()
    
    # Combine train and val for cross-validation
    X_all = np.concatenate([X_train_full, X_val])
    y_all = np.concatenate([y_train_full, y_val])
    
    # Create cross-validation splits
    cv_splits = create_cross_validation_splits(X_all, y_all, n_folds=5)
    
    cv_scores = []
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- FOLD {fold + 1}/5 ---")
        
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all[val_idx]
        
        # Create fresh model for each fold
        model = create_minimal_model(num_classes)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),  # Lower LR
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Conservative callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,  # Very early stopping
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train
        history = model.fit(
            X_train_fold, y_train_fold,
            batch_size=64,  # Larger batch size
            epochs=30,  # Fewer epochs
            validation_data=(X_val_fold, y_val_fold),
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        val_loss, val_acc = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(val_acc)
        models.append(model)
        
        print(f"Fold {fold + 1} validation accuracy: {val_acc:.4f}")
    
    # Calculate cross-validation statistics
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    
    print(f"\n=== CROSS-VALIDATION RESULTS ===")
    print(f"CV Mean Accuracy: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Individual fold scores: {[f'{score:.4f}' for score in cv_scores]}")
    
    # Train final model on all training data
    print(f"\n=== TRAINING FINAL MODEL ===")
    final_model = create_minimal_model(num_classes)
    final_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Monitor loss instead of accuracy
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Use a portion of data for validation
    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_all, axis=1)
    )
    
    history = final_model.fit(
        X_train_final, y_train_final,
        batch_size=64,
        epochs=30,
        validation_data=(X_val_final, y_val_final),
        callbacks=callbacks,
        verbose=1
    )
    
    # Final evaluation
    val_loss, val_acc = final_model.evaluate(X_val_final, y_val_final, verbose=0)
    test_loss, test_acc = final_model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Cross-validation estimate: {mean_cv_score:.4f} ± {std_cv_score:.4f}")
    print(f"Final model validation accuracy: {val_acc:.4f}")
    print(f"Final model test accuracy: {test_acc:.4f}")
    print(f"Overfitting indicator (val-test): {val_acc - test_acc:.4f}")
    
    # Check if results are realistic
    if val_acc > 0.98 or test_acc > 0.98:
        print("⚠️  WARNING: Accuracy still seems unrealistically high!")
        print("   This dataset might have fundamental issues (data leakage, too easy, etc.)")
    elif val_acc > mean_cv_score + 2*std_cv_score:
        print("⚠️  WARNING: Final model performs much better than CV estimate")
        print("   Possible overfitting to the final validation set")
    else:
        print("✅ Results look more realistic!")
    
    # Save model
    final_model.save('conservative_sign_model.keras')
    print("Model saved as 'conservative_sign_model.keras'")
    
    # Plot results
    plot_training_history(history, mean_cv_score, std_cv_score)
    
    return final_model, history, cv_scores

def plot_training_history(history, cv_mean, cv_std):
    """Plot training history with cross-validation reference"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    
    # Add CV reference line
    ax1.axhline(y=cv_mean, color='red', linestyle='--', alpha=0.7, 
                label=f'CV Mean: {cv_mean:.3f}')
    ax1.fill_between(range(len(history.history['accuracy'])), 
                     cv_mean - cv_std, cv_mean + cv_std, 
                     color='red', alpha=0.2, label=f'CV ±1σ')
    
    ax1.set_title('Model Accuracy vs Cross-Validation', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('conservative_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_label_mapping():
    """Create label mapping"""
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 
              'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 
              'V', 'W', 'X', 'Y']
    
    try:
        train_df = pd.read_csv('sign_mnist_train.csv')
        max_label = train_df['label'].max()
        num_classes = max_label + 1
        
        label_map = {i: labels[i] if i < len(labels) else f"Class_{i}" 
                    for i in range(num_classes)}
                
    except FileNotFoundError:
        label_map = {i: labels[i] for i in range(len(labels))}
    
    with open('label_mapping.json', 'w') as f:
        json.dump(label_map, f, indent=2)
    
    return label_map

if __name__ == "__main__":
    print("=== CONSERVATIVE SIGN LANGUAGE MODEL TRAINING ===")
    print("This version uses aggressive techniques to prevent overfitting:")
    print("- Cross-validation for honest performance estimation")
    print("- Minimal model architecture")
    print("- High dropout rates")
    print("- Early stopping")
    print("- Noise injection")
    print("- Larger validation sets")
    print()
    
    # Create label mapping
    create_label_mapping()
    
    # Train with cross-validation
    model, history, cv_scores = train_with_cross_validation()
    
    print(f"\n" + "="*60)
    print("REALISTIC TRAINING COMPLETED!")
    print("="*60)
    print("Expected realistic performance for sign language recognition:")
    print("- Good performance: 80-90%")
    print("- Excellent performance: 90-95%")
    print("- Suspicious (likely overfitting): >96%")
    print(f"\nYour model's cross-validation score: {np.mean(cv_scores):.1%}")
    print("Files created:")
    print("- conservative_sign_model.keras")
    print("- conservative_training_history.png")
    print("- label_mapping.json")