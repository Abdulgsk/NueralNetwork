import pickle
from tensorflow.keras.datasets import imdb
import numpy as np
import os
from scipy.sparse import csr_matrix

# --- Configuration ---
num_words = 5000  # Vocabulary size: consider the top 100,000 most frequent words
max_review_length = 200  # Optional: for padding if you choose that method later.

def load_and_preprocess_imdb_data(num_words_vocab=num_words):
    print(f"Loading IMDb data with vocab_size={num_words_vocab}...")
    (X_train_sequences, y_train_raw), (X_test_sequences, y_test_raw) = imdb.load_data(num_words=num_words_vocab)

    print(f"Found {len(X_train_sequences)} training sequences and {len(X_test_sequences)} test sequences.")

    # --- Vectorize sequences (Multi-hot encoding) ---
    print("Vectorizing sequences...")
    def vectorize_sequences(sequences, dimension):
        results = np.zeros((len(sequences), dimension), dtype=np.float32)
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.
        return results

    X_train = vectorize_sequences(X_train_sequences, num_words_vocab)
    X_test = vectorize_sequences(X_test_sequences, num_words_vocab)

    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of X_test: {X_test.shape}")

    # --- One-hot encode labels ---
    print("One-hot encoding labels...")
    y_train_oh = np.zeros((y_train_raw.size, 2))
    y_train_oh[np.arange(y_train_raw.size), y_train_raw] = 1

    y_test_oh = np.zeros((y_test_raw.size, 2))
    y_test_oh[np.arange(y_test_raw.size), y_test_raw] = 1

    print(f"Shape of y_train_oh: {y_train_oh.shape}")

    return X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw

def vectorize_single_review(review, num_words_vocab=5000):
    word_index = imdb.get_word_index()
    words = review.lower().split()
    indices = [word_index.get(word, 0) + 3 for word in words if word in word_index]
    indices = [i for i in indices if i < num_words_vocab and i >= 3]
    
    # Create a sparse vector
    if not indices:
        return np.zeros((num_words_vocab,), dtype=np.float32)
    data = np.ones(len(indices), dtype=np.float32)
    row_indices = np.zeros(len(indices), dtype=np.int32)
    col_indices = np.array(indices, dtype=np.int32)
    return csr_matrix((data, (row_indices, col_indices)), shape=(1, num_words_vocab)).toarray()


def decode_review(sequence_index, X_sequences_original, y_labels_original):
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in X_sequences_original[sequence_index]])
    sentiment = "Positive" if y_labels_original[sequence_index] == 1 else "Negative"
    
    print(f"\n--- Review Example ({sentiment}) ---")
    print(decoded_review)
    print(f"Label: {y_labels_original[sequence_index]}")

# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# --- Initialize weights and biases ---
def initialize_network(layer_sizes):
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        weights.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i]))
        biases.append(np.zeros((1, layer_sizes[i+1])))
    return weights, biases

# --- Forward pass ---
def forward_pass(x, weights, biases):
    activations = [x]
    for i in range(len(weights)):
        z = np.dot(activations[-1], weights[i]) + biases[i]
        if i == len(weights) - 1:
            a = softmax(z)
        else:
            a = relu(z)
        activations.append(a)
    return activations

# --- Cross-entropy loss ---
def compute_loss(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / len(y_true)

# --- Backward pass ---
def backward_pass(y_true, activations, weights):
    grads_w = [None] * len(weights)
    grads_b = [None] * len(weights)
    delta = activations[-1] - y_true
    for i in reversed(range(len(weights))):
        grads_w[i] = np.dot(activations[i].T, delta) / len(y_true)
        grads_b[i] = np.mean(delta, axis=0, keepdims=True)
        if i != 0:
            delta = np.dot(delta, weights[i].T) * relu_derivative(activations[i])
    return grads_w, grads_b

# --- Training loop ---
def train(X, y, layer_sizes, epochs=20, lr=0.5, batch_size=64):
    weights, biases = initialize_network(layer_sizes)
    for epoch in range(epochs):
        permutation = np.random.permutation(X.shape[0])
        X_shuffled = X[permutation]
        y_shuffled = y[permutation]

        for i in range(0, X.shape[0], batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            activations = forward_pass(X_batch, weights, biases)
            loss = compute_loss(y_batch, activations[-1])
            grads_w, grads_b = backward_pass(y_batch, activations, weights)

            for j in range(len(weights)):
                weights[j] -= lr * grads_w[j]
                biases[j] -= lr * grads_b[j]

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}")
    return weights, biases

def predict_probabilities(x, weights, biases):
    activations = forward_pass(x, weights, biases)
    return activations[-1]

def predict_classes(x, weights, biases):
    probabilities = predict_probabilities(x, weights, biases)
    return np.argmax(probabilities, axis=1)

def evaluate(X, y_true, weights, biases):
    preds = predict_classes(X, weights, biases)
    accuracy = np.mean(preds == y_true)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

def save_model(weights, biases, filename="model_params.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((weights, biases), f)

def load_model(filename="model_params.pkl"):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, 'rb') as f:
        weights, biases = pickle.load(f)
    return weights, biases

def interpret_sentiment_with_passage(review_vector, weights, biases):
    if review_vector.ndim == 1:
        review_vector = review_vector.reshape(1, -1)

    probabilities = predict_probabilities(review_vector, weights, biases)
    positive_prob = probabilities[0, 1]

    if positive_prob >= 0.99:
        sentiment_classification = "Cinematic Masterpiece!"
        descriptive_passage = "This movie is an absolute masterpiece..."
    elif positive_prob >= 0.95:
        sentiment_classification = "Must-Watch!"
        descriptive_passage = "A powerful and highly engrossing film..."
    elif positive_prob >= 0.88:
        sentiment_classification = "Very Positive"
        descriptive_passage = "This film offers a genuinely strong experience..."
    elif positive_prob >= 0.75:
        sentiment_classification = "Highly Recommended"
        descriptive_passage = "A genuinely enjoyable and well-crafted film..."
    elif positive_prob >= 0.60:
        sentiment_classification = "Generally Positive"
        descriptive_passage = "This film strikes a commendable balance..."
    elif positive_prob >= 0.50:
        sentiment_classification = "Slightly Positive"
        descriptive_passage = "This film has more positives than negatives..."
    elif positive_prob >= 0.40:
        sentiment_classification = "Mixed/Neutral"
        descriptive_passage = "The review suggests a mixed bag..."
    elif positive_prob >= 0.20:
        sentiment_classification = "Generally Negative"
        descriptive_passage = "This film appears to have significant flaws..."
    elif positive_prob >= 0.05:
        sentiment_classification = "Highly Disappointing"
        descriptive_passage = "A largely negative review..."
    else:
        sentiment_classification = "Absolute Worst!"
        descriptive_passage = "This is truly a film to avoid..."

    return sentiment_classification, descriptive_passage, positive_prob

# --- Entry point ---
if __name__ == "__main__":
    vocab_size = 5000
    X_train, y_train_oh, y_train_raw, X_test, y_test_oh, y_test_raw = load_and_preprocess_imdb_data(vocab_size)

    # Train the model
    layer_structure = [vocab_size, 128, 64, 2]  # Example: input -> hidden1 -> hidden2 -> output
    weights, biases = train(X_train, y_train_oh, layer_structure, epochs=10, lr=0.5, batch_size=128)

    # Evaluate on test set
    evaluate(X_test, y_test_raw, weights, biases)

    # Save the trained model
    save_model(weights, biases)

    # Interpret sentiment of a few sample reviews
    # for i in [0, 20, 30]:
    #     review_vector = X_test[i]
    #     sentiment_class, passage, prob = interpret_sentiment_with_passage(review_vector, weights, biases)
    #     print(f"\nReview {i + 1}:")
    #     print(f"Sentiment Classification: {sentiment_class}")
    #     print(f"Confidence (positive sentiment): {prob:.2f}")
    #     print(f"Interpretation: {passage}")
