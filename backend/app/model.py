import pickle
from tensorflow.keras.datasets import imdb
import numpy as np
import os
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# --- Configuration ---
num_words = 10000  # Increased vocabulary size for better feature representation
max_review_length = 200

def load_and_preprocess_imdb_data(num_words_vocab=num_words):
    print(f"Loading IMDb data with vocab_size={num_words_vocab}...")
    (X_train_sequences, y_train_raw), (X_test_sequences, y_test_raw) = imdb.load_data(num_words=num_words_vocab)

    print(f"Found {len(X_train_sequences)} training sequences and {len(X_test_sequences)} test sequences.")

    # --- Simple but effective vectorization ---
    print("Vectorizing sequences...")
    def vectorize_sequences(sequences, dimension):
        results = np.zeros((len(sequences), dimension), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            # Count word frequencies and normalize
            word_counts = {}
            for word_id in sequence:
                if word_id < dimension:
                    word_counts[word_id] = word_counts.get(word_id, 0) + 1
            
            # Set values based on word frequency (simple TF)
            for word_id, count in word_counts.items():
                results[i, word_id] = min(count, 5) / 5.0  # Cap at 5 and normalize
        
        return results

    X_train = vectorize_sequences(X_train_sequences, num_words_vocab)
    X_test = vectorize_sequences(X_test_sequences, num_words_vocab)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"X_train stats: mean={X_train.mean():.4f}, std={X_train.std():.4f}")

    # --- One-hot encode labels ---
    print("One-hot encoding labels...")
    y_train_oh = np.zeros((y_train_raw.size, 2))
    y_train_oh[np.arange(y_train_raw.size), y_train_raw] = 1

    y_test_oh = np.zeros((y_test_raw.size, 2))
    y_test_oh[np.arange(y_test_raw.size), y_test_raw] = 1

    print(f"Shape of y_train_oh: {y_train_oh.shape}")
    print(f"Label distribution - Positive: {y_train_raw.sum()}, Negative: {len(y_train_raw) - y_train_raw.sum()}")

    return X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw

def vectorize_single_review(review, num_words_vocab=10000):
    word_index = imdb.get_word_index()
    words = review.lower().split()
    indices = [word_index.get(word, 0) + 3 for word in words if word in word_index]
    indices = [i for i in indices if i < num_words_vocab and i >= 3]
    
    # Create weighted vector
    if not indices:
        return np.zeros((num_words_vocab,), dtype=np.float32)
    
    word_freq = {}
    for idx in indices:
        word_freq[idx] = word_freq.get(idx, 0) + 1
    
    result = np.zeros((num_words_vocab,), dtype=np.float32)
    for idx, freq in word_freq.items():
        result[idx] = np.log(freq + 1)  # TF weighting
    
    # Normalize
    norm = np.linalg.norm(result)
    if norm > 0:
        result = result / norm
    
    return result.reshape(1, -1)

def decode_review(sequence_index, X_sequences_original, y_labels_original):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_sequences_original[sequence_index]])
    sentiment = "Positive" if y_labels_original[sequence_index] == 1 else "Negative"
    
    print(f"\n--- Review Example ({sentiment}) ---")
    print(decoded_review)
    print(f"Label: {y_labels_original[sequence_index]}")

# --- Enhanced Activation functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# --- Enhanced weight initialization ---
def initialize_network(layer_sizes, activation='relu'):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        if activation == 'relu':
            # He initialization for ReLU - slightly smaller to prevent explosion
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.5 / layer_sizes[i]))
        else:
            # Xavier initialization
            weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1 / layer_sizes[i]))
        
        # Small positive bias for output layer to help with class balance
        if i == len(layer_sizes) - 2:  # Output layer
            biases.append(np.array([[0.1, -0.1]]))  # Slight positive bias for class 1
        else:
            biases.append(np.zeros((1, layer_sizes[i+1])))
    return weights, biases

# --- Enhanced forward pass with dropout ---
def forward_pass(x, weights, biases, training=True, dropout_rate=0.3):
    activations = [x]
    dropout_masks = []
    
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        
        if i == len(weights) - 1:  # Output layer
            a = softmax(z)
            dropout_masks.append(None)
        else:  # Hidden layers
            a = leaky_relu(z)  # Using Leaky ReLU for better gradient flow
            
            # Apply dropout during training
            if training and dropout_rate > 0:
                mask = np.random.binomial(1, 1 - dropout_rate, a.shape) / (1 - dropout_rate)
                a *= mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)
        
        activations.append(a)
    
    return activations, dropout_masks

# --- Cross-entropy loss with L2 regularization ---
def compute_loss(y_true, y_pred, weights, l2_reg=0.001):
    cross_entropy = -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)
    
    # L2 regularization
    l2_penalty = 0
    for w in weights:
        l2_penalty += np.sum(w ** 2)
    l2_penalty *= l2_reg / 2
    
    return cross_entropy + l2_penalty

# --- Enhanced backward pass ---
def backward_pass(y_true, activations, weights, dropout_masks, l2_reg=0.001):
    grads_w = [None] * len(weights)
    grads_b = [None] * len(weights)
    
    # Output layer gradient
    delta = activations[-1] - y_true
    
    for i in reversed(range(len(weights))):
        # Weight gradients with L2 regularization
        grads_w[i] = np.dot(activations[i].T, delta) / len(y_true) + l2_reg * weights[i]
        grads_b[i] = np.mean(delta, axis=0, keepdims=True)
        
        if i != 0:
            # Backpropagate through hidden layers
            delta = np.dot(delta, weights[i].T) * leaky_relu_derivative(activations[i])
            
            # Apply dropout mask if it was used during forward pass
            if dropout_masks[i-1] is not None:
                delta *= dropout_masks[i-1]
    
    return grads_w, grads_b

# --- Enhanced training loop with learning rate scheduling ---
def train(X, y, layer_sizes, epochs=50, initial_lr=0.01, batch_size=64, validation_split=0.2, 
          l2_reg=0.0001, dropout_rate=0.2, early_stopping_patience=15):
    
    # Split training data for validation
    val_size = int(len(X) * validation_split)
    indices = np.random.permutation(len(X))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    X_train_split = X[train_indices]
    y_train_split = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    weights, biases = initialize_network(layer_sizes)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None
    best_biases = None
    
    for epoch in range(epochs):
        # Learning rate scheduling (less aggressive decay)
        lr = initial_lr * (0.98 ** epoch)
        
        # Training
        permutation = np.random.permutation(X_train_split.shape[0])
        X_shuffled = X_train_split[permutation]
        y_shuffled = y_train_split[permutation]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train_split.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            activations, dropout_masks = forward_pass(X_batch, weights, biases, 
                                                    training=True, dropout_rate=dropout_rate)
            loss = compute_loss(y_batch, activations[-1], weights, l2_reg)
            grads_w, grads_b = backward_pass(y_batch, activations, weights, dropout_masks, l2_reg)
            
            # Update weights and biases
            for j in range(len(weights)):
                weights[j] -= lr * grads_w[j]
                biases[j] -= lr * grads_b[j]
            
            epoch_loss += loss
            num_batches += 1
        
        # Validation
        val_activations, _ = forward_pass(X_val, weights, biases, training=False)
        val_loss = compute_loss(y_val, val_activations[-1], weights, l2_reg)
        
        train_losses.append(epoch_loss / num_batches)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, "
              f"Val Loss: {val_loss:.4f}, LR: {lr:.6f}")
        
        # Print sample predictions every 10 epochs for debugging
        if (epoch + 1) % 10 == 0:
            sample_preds = predict_probabilities(X_val[:100], weights, biases)
            pos_preds = np.sum(sample_preds[:, 1] > 0.5)
            print(f"  Sample predictions: {pos_preds}/100 classified as positive")
        
        # Early stopping
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
    
    return weights, biases, train_losses, val_losses

def predict_probabilities(x, weights, biases):
    activations, _ = forward_pass(x, weights, biases, training=False)
    return activations[-1]

def predict_classes(x, weights, biases):
    probabilities = predict_probabilities(x, weights, biases)
    return np.argmax(probabilities, axis=1)

def evaluate(X, y_true, weights, biases, detailed=True):
    preds = predict_classes(X, weights, biases)
    accuracy = np.mean(preds == y_true)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    if detailed:
        print("\nDetailed Classification Report:")
        print(classification_report(y_true, preds, target_names=['Negative', 'Positive']))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, preds))
    
    return accuracy

def save_model(weights, biases, filename="enhanced_model_params.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((weights, biases), f)
    print(f"Model saved to {filename}")

def load_model(filename="enhanced_model_params.pkl"):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'rb') as f:
        weights, biases = pickle.load(f)
    return weights, biases

def interpret_sentiment_with_passage(review_vector, weights, biases):
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = predict_probabilities(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]
    confidence = max(positive_prob, 1 - positive_prob)  # Confidence in the prediction

    if positive_prob >= 0.95:
        sentiment_classification = "Overwhelmingly Positive"
        descriptive_passage = "This movie delivers an exceptional cinematic experience with outstanding performances, compelling storytelling, and masterful direction that resonates deeply with audiences."
    elif positive_prob >= 0.85:
        sentiment_classification = "Highly Positive"
        descriptive_passage = "A genuinely impressive film that excels in multiple aspects, offering strong entertainment value and memorable moments that justify enthusiastic recommendation."
    elif positive_prob >= 0.70:
        sentiment_classification = "Positive"
        descriptive_passage = "This film succeeds in delivering an engaging and well-crafted experience with solid performances and effective storytelling that entertains throughout."
    elif positive_prob >= 0.55:
        sentiment_classification = "Moderately Positive"
        descriptive_passage = "While not without minor flaws, this film offers more strengths than weaknesses with decent execution and moments of genuine quality."
    elif positive_prob >= 0.45:
        sentiment_classification = "Neutral/Mixed"
        descriptive_passage = "This film presents a balanced mix of positive and negative elements, resulting in an average viewing experience with both highlights and disappointments."
    elif positive_prob >= 0.30:
        sentiment_classification = "Moderately Negative"
        descriptive_passage = "Despite some redeeming qualities, this film suffers from notable issues in execution, pacing, or storytelling that detract from the overall experience."
    elif positive_prob >= 0.15:
        sentiment_classification = "Negative"
        descriptive_passage = "This film fails to deliver on multiple fronts with significant problems in direction, performance, or script that make it difficult to recommend."
    elif positive_prob >= 0.05:
        sentiment_classification = "Highly Negative"
        descriptive_passage = "A largely unsuccessful film with major flaws that severely impact the viewing experience, offering little in terms of entertainment or artistic value."
    else:
        sentiment_classification = "Overwhelmingly Negative"
        descriptive_passage = "This film represents a significant misfire with fundamental problems across all aspects of production, resulting in a thoroughly disappointing experience."

    return sentiment_classification, descriptive_passage, positive_prob, confidence

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Entry point ---
if __name__ == "__main__":
    vocab_size = 10000
    X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw = load_and_preprocess_imdb_data(vocab_size)

    # Simplified but effective architecture - deeper networks can be harder to train
    layer_structure = [vocab_size, 256, 128, 2]  # Simpler, more stable architecture
    
    print("Training enhanced model...")
    weights, biases, train_losses, val_losses = train(
        X_train, y_train_oh, layer_structure, 
        epochs=50, 
        initial_lr=0.01,  # Higher learning rate
        batch_size=64,    # Smaller batch size for better gradients
        l2_reg=0.0001,    # Reduced regularization
        dropout_rate=0.2, # Reduced dropout
        early_stopping_patience=15
    )

    # Evaluate on test set
    print("\n" + "="*50)
    print("FINAL MODEL EVALUATION")
    print("="*50)
    test_accuracy = evaluate(X_test, y_test_raw, weights, biases, detailed=True)

    # Save the trained model
    save_model(weights, biases)

    # Plot training history
    try:
        plot_training_history(train_losses, val_losses)
    except:
        print("Could not display training plots (matplotlib may not be available)")

    # Interpret sentiment of sample reviews with enhanced analysis
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    for i in [0, 100, 500, 1000, 2000]:
        if i < len(X_test):
            review_vector = X_test[i]
            sentiment_class, passage, prob, confidence = interpret_sentiment_with_passage(
                review_vector, weights, biases
            )
            actual_sentiment = "Positive" if y_test_raw[i] == 1 else "Negative"
            
            print(f"\nSample Review {i + 1}:")
            print(f"Actual Sentiment: {actual_sentiment}")
            print(f"Predicted Sentiment: {sentiment_class}")
            print(f"Confidence: {confidence:.3f}")
            print(f"Positive Probability: {prob:.3f}")
            print(f"Analysis: {passage}")
            print("-" * 80)

    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")