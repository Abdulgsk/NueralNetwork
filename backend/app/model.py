import pickle
from tensorflow.keras.datasets import imdb
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import time
import re

# --- Configuration ---
num_words = 15000
max_review_length = 300

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\!\?\,\-\']', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def load_and_preprocess_imdb_data(num_words_vocab=num_words):
    print(f"Loading IMDb data with vocab_size={num_words_vocab}...")
    start_time = time.time()
    
    (X_train_sequences, y_train_raw), (X_test_sequences, y_test_raw) = imdb.load_data(num_words=num_words_vocab)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    print(f"Found {len(X_train_sequences)} training sequences and {len(X_test_sequences)} test sequences.")

    # --- Improved TF-IDF vectorization ---
    print("Computing document frequencies...")
    start_time = time.time()
    
    # Calculate document frequencies more efficiently
    doc_freqs = np.zeros(num_words_vocab, dtype=np.int32)
    total_docs = len(X_train_sequences)
    
    for i, sequence in enumerate(X_train_sequences):
        if i % 5000 == 0:
            print(f"Processing document {i}/{total_docs}")
        
        unique_words = set(word_id for word_id in sequence if word_id < num_words_vocab)
        for word_id in unique_words:
            doc_freqs[word_id] += 1
    
    # Calculate IDF values with smoother scaling
    idf_values = np.log(total_docs / (doc_freqs + 1)) + 1  # Add 1 to prevent negative values
    print(f"IDF computation completed in {time.time() - start_time:.2f} seconds")

    def vectorize_sequences_improved(sequences, dimension, name="sequences"):
        print(f"Vectorizing {name}...")
        start_time = time.time()
        
        results = np.zeros((len(sequences), dimension), dtype=np.float32)
        
        for i, sequence in enumerate(sequences):
            if i % 5000 == 0:
                print(f"Vectorizing {name}: {i}/{len(sequences)} ({i/len(sequences)*100:.1f}%)")
            
            # Count word frequencies
            word_counts = Counter(word_id for word_id in sequence if word_id < dimension)
            
            # Apply improved TF-IDF with better scaling
            for word_id, count in word_counts.items():
                # Use log normalization for TF
                tf = 1 + np.log(count)  # This prevents zero values
                results[i, word_id] = tf * idf_values[word_id]
            
            # Use L2 normalization but with better scaling
            norm = np.linalg.norm(results[i])
            if norm > 0:
                results[i] = results[i] / norm
        
        print(f"{name} vectorization completed in {time.time() - start_time:.2f} seconds")
        return results

    X_train = vectorize_sequences_improved(X_train_sequences, num_words_vocab, "training sequences")
    X_test = vectorize_sequences_improved(X_test_sequences, num_words_vocab, "test sequences")

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"X_train stats: mean={X_train.mean():.6f}, std={X_train.std():.6f}")

    # --- One-hot encode labels ---
    print("One-hot encoding labels...")
    y_train_oh = np.eye(2)[y_train_raw]
    y_test_oh = np.eye(2)[y_test_raw]

    print(f"Shape of y_train_oh: {y_train_oh.shape}")
    print(f"Label distribution - Positive: {y_train_raw.sum()}, Negative: {len(y_train_raw) - y_train_raw.sum()}")

    return X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw, idf_values

def vectorize_single_review(review, idf_values, num_words_vocab=15000):
    """Improved vectorization for single review"""
    # Get word index
    word_index = imdb.get_word_index()
    
    # Preprocess the review text
    review = preprocess_text(review)
    words = review.split()
    
    # Convert words to indices with proper offset (IMDb uses +3 offset)
    indices = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3  # IMDb uses +3 offset
            if 3 <= idx < num_words_vocab:  # Valid range
                indices.append(idx)
    
    if not indices:
        print("Warning: No valid words found in review")
        return np.zeros((num_words_vocab,), dtype=np.float32)
    
    # Calculate TF with improved method
    word_counts = Counter(indices)
    result = np.zeros((num_words_vocab,), dtype=np.float32)
    
    for idx, freq in word_counts.items():
        if idx < len(idf_values):
            # Improved TF calculation
            tf = 1 + np.log(freq)  # Prevents zero values
            result[idx] = tf * idf_values[idx]
    
    # L2 normalization
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    
    return result.reshape(1, -1)

# --- Improved activation functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    # More stable softmax implementation
    x_shifted = x - np.max(x, axis=1, keepdims=True)
    e_x = np.exp(np.clip(x_shifted, -500, 500))
    return e_x / (np.sum(e_x, axis=1, keepdims=True) + 1e-8)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# --- Improved network initialization ---
def initialize_network(layer_sizes):
    weights = []
    biases = []
    
    for i in range(len(layer_sizes) - 1):
        # Xavier/Glorot initialization for better gradient flow
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights.append(np.random.normal(0, std, (layer_sizes[i], layer_sizes[i+1])))
        biases.append(np.zeros((1, layer_sizes[i+1])))
    
    return weights, biases

# --- Forward pass remains the same ---
def forward_pass(x, weights, biases, training=True, dropout_rate=0.2):
    activations = [x]
    dropout_masks = []
    
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if i == len(weights) - 1:  # Output layer
            a = softmax(z)
            dropout_masks.append(None)
        else:  # Hidden layers
            a = leaky_relu(z)
            
            # Apply dropout during training
            if training and dropout_rate > 0:
                mask = np.random.binomial(1, 1 - dropout_rate, a.shape) / (1 - dropout_rate)
                a *= mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
        
        activations.append(a)
    
    return activations, dropout_masks

# --- Improved loss function ---
def compute_loss(y_true, y_pred, weights, l2_reg=0.0001):
    # Cross-entropy loss with better clipping
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
    cross_entropy = -np.sum(y_true * np.log(y_pred_clipped)) / len(y_true)
    
    # L2 regularization
    l2_penalty = sum(np.sum(w ** 2) for w in weights) * l2_reg / 2
    
    return cross_entropy + l2_penalty

# --- Backward pass remains the same ---
def backward_pass(y_true, activations, weights, dropout_masks, l2_reg=0.0001):
    grads_w = [None] * len(weights)
    grads_b = [None] * len(weights)
    
    # Output layer gradient
    delta = activations[-1] - y_true
    
    for i in reversed(range(len(weights))):
        # Gradient clipping
        delta = np.clip(delta, -10, 10)  # Increased clipping range
        
        # Weight and bias gradients
        grads_w[i] = np.dot(activations[i].T, delta) / len(y_true) + l2_reg * weights[i]
        grads_b[i] = np.mean(delta, axis=0, keepdims=True)
        
        if i != 0:
            # Backpropagate through hidden layers
            delta = np.dot(delta, weights[i].T) * leaky_relu_derivative(activations[i])
            
            # Apply dropout mask
            if dropout_masks[i-1] is not None:
                delta *= dropout_masks[i-1]
    
    return grads_w, grads_b

# --- Improved training function ---
def train(X, y, layer_sizes, epochs=100, initial_lr=0.003, batch_size=64, validation_split=0.15, 
          l2_reg=0.0001, dropout_rate=0.1, early_stopping_patience=20):
    
    print("Splitting training data...")
    # Split training data for validation
    val_size = int(len(X) * validation_split)
    indices = np.random.permutation(len(X))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train_split = X[train_indices]
    y_train_split = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    print(f"Training samples: {len(X_train_split)}, Validation samples: {len(X_val)}")
    
    weights, biases = initialize_network(layer_sizes)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    best_biases = None
    
    print("Starting training...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Improved learning rate decay
        lr = initial_lr * (0.96 ** (epoch // 10))
        
        # Shuffle training data
        permutation = np.random.permutation(X_train_split.shape[0])
        X_shuffled = X_train_split[permutation]
        y_shuffled = y_train_split[permutation]
        
        epoch_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, X_train_split.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward pass
            activations, dropout_masks = forward_pass(X_batch, weights, biases, 
                                                    training=True, dropout_rate=dropout_rate)
            loss = compute_loss(y_batch, activations[-1], weights, l2_reg)
            
            # Backward pass
            grads_w, grads_b = backward_pass(y_batch, activations, weights, dropout_masks, l2_reg)
            
            # Update weights and biases with gradient clipping
            for j in range(len(weights)):
                # Clip gradients
                grads_w[j] = np.clip(grads_w[j], -5, 5)
                grads_b[j] = np.clip(grads_b[j], -5, 5)
                
                weights[j] -= lr * grads_w[j]
                biases[j] -= lr * grads_b[j]
            
            epoch_loss += loss
            num_batches += 1
        
        # Validation
        val_activations, _ = forward_pass(X_val, weights, biases, training=False)
        val_loss = compute_loss(y_val, val_activations[-1], weights, l2_reg)
        
        # Calculate validation accuracy
        val_preds = np.argmax(val_activations[-1], axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_acc = np.mean(val_preds == val_true)
        
        train_losses.append(epoch_loss / num_batches)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1:3d}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {lr:.6f}, "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping with improved patience
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_weights = [w.copy() for w in weights]
            best_biases = [b.copy() for b in biases]
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                weights = best_weights
                biases = best_biases
                break
    
    return weights, biases, train_losses, val_losses, val_accuracies

def predict_probabilities(x, weights, biases):
    activations, _ = forward_pass(x, weights, biases, training=False)
    return activations[-1]

def predict_classes(x, weights, biases):
    probabilities = predict_probabilities(x, weights, biases)
    return np.argmax(probabilities, axis=1)

def evaluate(X, y_true, weights, biases, detailed=True):
    print("Evaluating model...")
    preds = predict_classes(X, weights, biases)
    accuracy = np.mean(preds == y_true)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    if detailed:
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, preds, target_names=['Negative', 'Positive']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, preds))
    
    return accuracy

def save_model(weights, biases, idf_values, filename="sentiment_model_improved.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((weights, biases, idf_values), f)
    print(f"Model saved to {filename}")

def load_model(filename="sentiment_model_improved.pkl"):
    with open(filename, 'rb') as f:
        weights, biases, idf_values = pickle.load(f)
    return weights, biases, idf_values

def interpret_sentiment_with_passage(review_vector, weights, biases):
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = predict_probabilities(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]
    confidence = abs(positive_prob - 0.5) * 2

    # More balanced thresholds
    if positive_prob >= 0.9:
        sentiment_classification = "Overwhelmingly Positive"
        descriptive_passage = "This movie delivers an exceptional cinematic experience with outstanding performances, compelling storytelling, and masterful direction."
    elif positive_prob >= 0.8:
        sentiment_classification = "Highly Positive" 
        descriptive_passage = "A genuinely impressive film that excels in multiple aspects, offering strong entertainment value and memorable moments."
    elif positive_prob >= 0.65:
        sentiment_classification = "Positive"
        descriptive_passage = "This film succeeds in delivering an engaging and well-crafted experience with solid performances and effective storytelling."
    elif positive_prob >= 0.55:
        sentiment_classification = "Moderately Positive"
        descriptive_passage = "While not perfect, this film offers more strengths than weaknesses with decent execution and genuine quality moments."
    elif positive_prob >= 0.45:
        sentiment_classification = "Neutral/Mixed"
        descriptive_passage = "This film presents a balanced mix of positive and negative elements, resulting in an average viewing experience."
    elif positive_prob >= 0.35:
        sentiment_classification = "Moderately Negative"
        descriptive_passage = "Despite some redeeming qualities, this film suffers from notable issues that detract from the overall experience."
    elif positive_prob >= 0.2:
        sentiment_classification = "Negative"
        descriptive_passage = "This film fails to deliver on multiple fronts with significant problems that make it difficult to recommend."
    elif positive_prob >= 0.1:
        sentiment_classification = "Highly Negative"
        descriptive_passage = "A largely unsuccessful film with major flaws that severely impact the viewing experience."
    else:
        sentiment_classification = "Overwhelmingly Negative"
        descriptive_passage = "This film represents a significant misfire with fundamental problems across all aspects of production."

    return sentiment_classification, descriptive_passage, positive_prob, confidence

def debug_review_vectorization(review, idf_values, num_words_vocab=15000):
    """Debug function to understand how review is being processed"""
    word_index = imdb.get_word_index()
    
    print(f"Original review: {review}")
    processed_review = preprocess_text(review)
    print(f"Processed review: {processed_review}")
    
    words = processed_review.split()
    print(f"Words: {words[:10]}...")  # Show first 10 words
    
    indices = []
    found_words = []
    for word in words:
        if word in word_index:
            idx = word_index[word] + 3
            if 3 <= idx < num_words_vocab:
                indices.append(idx)
                found_words.append(word)
    
    print(f"Found {len(found_words)} valid words out of {len(words)} total words")
    print(f"First 10 found words: {found_words[:10]}")
    print(f"Corresponding indices: {indices[:10]}")
    
    return vectorize_single_review(review, idf_values, num_words_vocab)

def plot_training_history(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='red', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='green', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(np.diff(val_losses), label='Val Loss Change', color='purple', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Change')
    plt.title('Validation Loss Change Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --- Entry point ---
if __name__ == "__main__":
    print("="*60)
    print("IMPROVED IMDB SENTIMENT ANALYSIS MODEL")
    print("="*60)
    
    vocab_size = 15000
    
    # Load and preprocess data
    total_start = time.time()
    X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw, idf_values = load_and_preprocess_imdb_data(vocab_size)
    
    # Improved model architecture
    layer_structure = [vocab_size, 512, 256, 128, 2]  # Deeper network for better learning
    print(f"\nModel architecture: {layer_structure}")
    
    # Train model with improved hyperparameters
    print("\nTraining model...")
    weights, biases, train_losses, val_losses, val_accuracies = train(
        X_train, y_train_oh, layer_structure, 
        epochs=100,
        initial_lr=0.003,  # Lower learning rate
        batch_size=64,
        l2_reg=0.0001,
        dropout_rate=0.1,  # Lower dropout
        early_stopping_patience=20
    )

    # Evaluate on test set
    print("\n" + "="*60)
    print("FINAL MODEL EVALUATION")
    print("="*60)
    test_accuracy = evaluate(X_test, y_test_raw, weights, biases, detailed=True)

    # Save the trained model
    save_model(weights, biases, idf_values, "sentiment_model_improved.pkl")
    
    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time/60:.1f} minutes")

    # Test with sample reviews with debugging
    print("\n" + "="*60)
    print("TESTING WITH SAMPLE REVIEWS (WITH DEBUGGING)")
    print("="*60)
    
    # Test The Shawshank Redemption review
    shawshank_review = """The Shawshank Redemption is absolutely incredible and a true masterpiece of cinema. 
    This movie is deeply moving and brilliantly told with outstanding performances by Tim Robbins and Morgan Freeman. 
    The direction by Frank Darabont is superb and every scene serves the story perfectly. 
    The themes of hope, friendship, and redemption are beautifully woven throughout the narrative. 
    This is genuinely one of the greatest films ever made and deserves all the praise and recognition it receives. 
    Absolutely phenomenal storytelling with perfect pacing and emotional depth."""
    
    print("Testing The Shawshank Redemption review:")
    print(f"Review: {shawshank_review}")
    
    review_vector = debug_review_vectorization(shawshank_review, idf_values, vocab_size)
    sentiment_class, passage, prob, confidence = interpret_sentiment_with_passage(review_vector, weights, biases)
    
    print(f"\nPredicted Sentiment: {sentiment_class}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Positive Probability: {prob:.3f}")
    print(f"Analysis: {passage}")
    
    # Test negative review
    print("\n" + "-"*50)
    negative_review = """This movie was absolutely terrible and a complete waste of time and money. 
    The acting was horrible and unconvincing, the plot made absolutely no sense whatsoever, and the direction was completely amateurish. 
    I couldn't wait for this boring disaster to finally end. One of the worst films I have ever had the misfortune to watch."""
    
    print("Testing negative review:")
    print(f"Review: {negative_review}")
    
    review_vector = debug_review_vectorization(negative_review, idf_values, vocab_size)
    sentiment_class, passage, prob, confidence = interpret_sentiment_with_passage(review_vector, weights, biases)
    
    print(f"\nPredicted Sentiment: {sentiment_class}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Positive Probability: {prob:.3f}")
    print(f"Analysis: {passage}")

    # Test with the actual Godfather reviews from your document
    print("\n" + "-"*50)
    godfather_review = """This film works so well because it takes place in an underworld in which we are so embedded that we do not even observe it. 
    This is a film of unmatched subtlety. No other movie sustains itself as good. No other film is done with such precision, attention and completeness. 
    What director Francis Ford Coppola did is nothing short of a timeless piece of reference cinema whose influence is not based on reinventing the wheel, 
    but rather perfecting it to the absolute maximum. This is one of the few films that will be remembered simply because they are that good and 
    I cannot possibly imagine a greater achievement."""
    
    print("Testing Godfather review from your document:")
    print(f"Review: {godfather_review}")
    
    review_vector = debug_review_vectorization(godfather_review, idf_values, vocab_size)
    sentiment_class, passage, prob, confidence = interpret_sentiment_with_passage(review_vector, weights, biases)
    
    print(f"\nPredicted Sentiment: {sentiment_class}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Positive Probability: {prob:.3f}")
    print(f"Analysis: {passage}")

    # Plot training history
    try:
        plot_training_history(train_losses, val_losses, val_accuracies)
    except Exception as e:
        print(f"Could not display plots: {e}")

    print(f"\n" + "="*60)
    print(f"FINAL RESULTS")
    print(f"="*60)
    print(f"Final Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Training completed successfully!")