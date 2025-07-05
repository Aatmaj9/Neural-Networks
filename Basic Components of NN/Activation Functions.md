While building a neural network, one key decision is selecting the Activation Function for both the hidden layer and the output layer. It is a mathematical function applied to the output of a neuron. It introduces non-linearity into the model, allowing the network to learn and represent complex patterns in the data. Without this non-linearity feature a neural network would behave like a linear regression model no matter how many layers it has.

# Types of Activation Functions in Deep Learning

## Linear Activation Function
Linear Activation Function resembles straight line define by y=x. No matter how many layers the neural network contains if they all use linear activation functions the output is a linear combination of the input.

The range of the output spans from 
(
−
∞
 to 
+
∞
)
(−∞ to +∞).
Linear activation function is used at just one place i.e. output layer.
Using linear activation across all layers makes the network's ability to learn complex patterns limited.
Linear activation functions are useful for specific tasks but must be combined with non-linear functions to enhance the neural network’s learning and predictive capabilities.

Linear-Activation-Function
Linear Activation Function or Identity Function returns the input as the output
## Non-Linear Activation Functions
### 1. Sigmoid Function 
### 2. tanh Function

Advantages of Using Tanh

Symmetry Around Zero: Since the output is centered around zero, the network has a better chance of balancing the weights. This helps in ensuring that the gradients don't just keep increasing or decreasing in magnitude, making training faster and more stable.

Improved Convergence: The tanh function is differentiable, making it a good candidate for training deep networks using gradient-based optimization algorithms like stochastic gradient descent (SGD).

Gradient Descent Efficiency: Unlike the sigmoid, which is constrained between 0 and 1, the tanh function’s output between -1 and 1 helps in better weight updates during training, leading to improved convergence speed.

Disadvantages of Tanh

Vanishing Gradient Problem: Similar to the sigmoid function, tanh suffers from the vanishing gradient problem for large values of the input (both positive and negative). When the input to the tanh function is very large or very small, the gradient approaches zero, which can slow down or halt learning during backpropagation, especially in deep networks.

Not Suitable for All Tasks: While tanh works well in many cases, it might not be the best option for all types of neural network architectures. For instance, ReLU (Rectified Linear Unit) has gained popularity for deep networks due to its simplicity and efficiency in mitigating the vanishing gradient problem.

Sensitive to Outliers: Extreme values in the input can lead to saturated regions where the gradient is close to zero, making learning slow or ineffective. This could happen if the inputs to the tanh function are not scaled properly.

When to Use Tanh?

The data is already normalized or centered around zero.
You are building shallow neural networks (i.e., networks with fewer layers).
You are working with data where negative values are significant and should be retained.
### 3. ReLU Function

## Exponential Linear Units

### 1. Softmax Funciton

### 2. Softplus Function
