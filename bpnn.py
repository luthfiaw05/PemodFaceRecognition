import numpy as np
import pickle

class BPNN:
    """Backpropagation Neural Network for Face Recognition"""
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input neurons (flattened image pixels)
            hidden_size: Number of hidden layer neurons
            output_size: Number of output neurons (number of classes/people)
            learning_rate: Learning rate for weight updates
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_ih = np.random.randn(input_size, hidden_size) * 0.01
        self.weights_ho = np.random.randn(hidden_size, output_size) * 0.01
        
        self.bias_h = np.zeros((1, hidden_size))
        self.bias_o = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def softmax(self, x):
        """Softmax activation for output layer"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (batch_size, input_size)
            
        Returns:
            output: Network output
        """
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_ih) + self.bias_h
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Output layer
        self.output_input = np.dot(self.hidden_output, self.weights_ho) + self.bias_o
        self.output = self.softmax(self.output_input)
        
        return self.output
    
    def backward(self, X, y, output):
        """
        Backward propagation
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            output: Network output from forward pass
        """
        m = X.shape[0]  
        
        output_error = output - y
        output_delta = output_error
        
        hidden_error = np.dot(output_delta, self.weights_ho.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        self.weights_ho -= self.learning_rate * np.dot(self.hidden_output.T, output_delta) / m
        self.bias_o -= self.learning_rate * np.sum(output_delta, axis=0, keepdims=True) / m
        
        self.weights_ih -= self.learning_rate * np.dot(X.T, hidden_delta) / m
        self.bias_h -= self.learning_rate * np.sum(hidden_delta, axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs=1000, batch_size=32, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training data
            y: Training labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Print training progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            total_loss = 0
            
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                output = self.forward(batch_X)
                
                loss = -np.sum(batch_y * np.log(output + 1e-8)) / batch_X.shape[0]
                total_loss += loss
                
                self.backward(batch_X, batch_y, output)
            
            if verbose and (epoch + 1) % 100 == 0:
                avg_loss = total_loss / (n_samples / batch_size)
                accuracy = self.evaluate(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
            
        Returns:
            predictions: Predicted class indices
        """
        output = self.forward(X)
        return np.argmax(output, axis=1)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input data
            
        Returns:
            probabilities: Prediction probabilities for each class
        """
        return self.forward(X)
    
    def evaluate(self, X, y):
        """
        Evaluate model accuracy
        
        Args:
            X: Input data
            y: True labels (one-hot encoded)
            
        Returns:
            accuracy: Accuracy percentage
        """
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == true_labels) * 100
        return accuracy
    
    def save_model(self, filename):
        """Save model to file"""
        model_data = {
            'weights_ih': self.weights_ih,
            'weights_ho': self.weights_ho,
            'bias_h': self.bias_h,
            'bias_o': self.bias_o,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load model from file"""
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.weights_ih = model_data['weights_ih']
        self.weights_ho = model_data['weights_ho']
        self.bias_h = model_data['bias_h']
        self.bias_o = model_data['bias_o']
        self.input_size = model_data['input_size']
        self.hidden_size = model_data['hidden_size']
        self.output_size = model_data['output_size']
        self.learning_rate = model_data['learning_rate']
        print(f"Model loaded from {filename}")