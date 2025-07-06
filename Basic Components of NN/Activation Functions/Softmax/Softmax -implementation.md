**Step 1: Import Required Libraries**

We need numpy for matrix operations and numerical computations, as it handles operations on arrays.

```
import numpy as np
```

**Step 2: Define Activation Functions**

We define two key activation functions for the network: softmax for the output layer and relu for the hidden layer.

```
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Numerical stability improvement
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

**Step 3: Initialize the Neural Network**

In this step, we define the structure of our neural network:

input_size: Number of input features.
hidden_size: Number of neurons in the hidden layer.
output_size: Number of output classes for multi-class classification.
The weights (W1, W2) and biases (b1, b2) are initialized for the hidden and output layers.

```

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Weights for hidden layer
        self.b1 = np.zeros((1, hidden_size))  # Biases for hidden layer
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Weights for output layer
        self.b2 = np.zeros((1, output_size))  # Biases for output layer
```

**Step 4: Forward Pass**

This step computes the output of the neural network by passing data through two layers:

Layer 1: The input is passed through the hidden layer, using the ReLU activation function.
Layer 2: The output from the hidden layer is passed through the output layer, using the softmax activation function.

```
def forward(self, X):
    self.Z1 = np.dot(X, self.W1) + self.b1  # Linear combination for hidden layer
    self.A1 = relu(self.Z1)  # ReLU activation in hidden layer
    
    self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Linear combination for output layer
    self.A2 = softmax(self.Z2)  # Softmax activation for output layer (probabilities)
    
    return self.A2  # Return predicted probabilities
```

**Step 5: Loss Function (Cross-Entropy)**

The loss is computed by comparing the predicted probabilities (Y_hat) with the actual labels (Y). The cross-entropy loss function is used, which penalizes wrong predictions.

```
def compute_loss(self, Y, Y_hat):
    m = Y.shape[0]  # Number of samples
    log_likelihood = -np.log(Y_hat[range(m), Y])  # Select the predicted probability for the correct class
    loss = np.sum(log_likelihood) / m  # Average cross-entropy loss
    return loss
```

