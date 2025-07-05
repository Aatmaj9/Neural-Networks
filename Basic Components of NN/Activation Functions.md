While building a neural network, one key decision is selecting the Activation Function for both the hidden layer and the output layer. It is a mathematical function applied to the output of a neuron. It introduces non-linearity into the model, allowing the network to learn and represent complex patterns in the data. Without this non-linearity feature a neural network would behave like a linear regression model no matter how many layers it has.

# Types of Activation Functions in Deep Learning

## Linear Activation Function

Linear Activation Function resembles straight line define by y=x. No matter how many layers the neural network contains if they all use linear activation functions the output is a linear combination of the input.
The range of the output spans from (−∞ to +∞).
Linear activation function is used at just one place i.e. output layer.
Using linear activation across all layers makes the network's ability to learn complex patterns limited.
Linear activation functions are useful for specific tasks but must be combined with non-linear functions to enhance the neural network’s learning and predictive capabilities.

![image](https://github.com/user-attachments/assets/df8cd303-6fb6-4e70-923f-ad6c790108a8)

## Non-Linear Activation Functions
### 1. Sigmoid Function 

Sigmoid is a mathematical function that maps any real-valued number into a value between 0 and 1.

σ= 1/(1+e^(−x))
​
**Properties of the Sigmoid Function**

The sigmoid function has several key properties that make it a popular choice in machine learning and neural networks:

Domain: The domain of the sigmoid function is all real numbers. This means that you can input any real number into the sigmoid function, and it will produce a valid output.

Asymptotes: As x approaches positive infinity, σ(x) approaches 1. Conversely, as x approaches negative infinity, σ(x) approaches 0. This property ensures that the function never actually reaches 0 or 1, but gets arbitrarily close.

Monotonicity: The sigmoid function is monotonically increasing, meaning that as the input increases, the output also increases.
Differentiability: The sigmoid function is differentiable, which allows for the calculation of gradients during the training of machine learning models.

![image](https://github.com/user-attachments/assets/b0e10922-84b6-497f-a424-852f68b9b19b)

**SIgmoid function in back-propagation**

During the backpropagation, the model calculates and updates weights and biases by computing the derivative of the loss function which involves the activation function. The sigmoid function is useful because:

1. It is the only function that appears in its derivative.

2. It is differentiable at every point, which helps in the effective computation of gradients during backpropagation.

One key issue with using the sigmoid function is the vanishing gradient problem. When updating weights and biases using gradient descent, if the gradients are too small, the updates to weights and biases become insignificant, slowing down or even stopping learning.

![image](https://github.com/user-attachments/assets/07727582-8507-4636-aab7-502c740c8cdc)


### 2. tanh Function

Tanh (hyperbolic tangent) is a type of activation function that transforms its input into a value between -1 and 1.

![image](https://github.com/user-attachments/assets/70ae5b14-c966-4cc4-8e3c-5667e01d782c)

**Advantages of Using Tanh**

Non-linearity: Tanh introduces non-linearity to the model, which allows neural networks to learn complex patterns and relationships in the data. Without non-linear activation functions, a neural network would essentially behave as a linear model, no matter how many layers it has.

Centered Around Zero: The output of the tanh function is centered around 0, unlike the sigmoid function, which outputs values between 0 and 1. This makes the tanh activation function more useful for many types of tasks, as the mean of the output is closer to zero, leading to more efficient training and faster convergence.

Gradient Behavior: Tanh helps mitigate the vanishing gradient problem (to some extent), especially when compared to sigmoid activation. This is because the gradient of the tanh function is generally higher than that of the sigmoid, enabling better weight updates during backpropagation.

**Disadvantages of Tanh**

Vanishing Gradient Problem: Similar to the sigmoid function, tanh suffers from the vanishing gradient problem for large values of the input (both positive and negative). When the input to the tanh function is very large or very small, the gradient approaches zero, which can slow down or halt learning during backpropagation, especially in deep networks.

Not Suitable for All Tasks: While tanh works well in many cases, it might not be the best option for all types of neural network architectures. For instance, ReLU (Rectified Linear Unit) has gained popularity for deep networks due to its simplicity and efficiency in mitigating the vanishing gradient problem.

Sensitive to Outliers: Extreme values in the input can lead to saturated regions where the gradient is close to zero, making learning slow or ineffective. This could happen if the inputs to the tanh function are not scaled properly.

**When to Use Tanh?**

The data is already normalized or centered around zero.
You are building shallow neural networks (i.e., networks with fewer layers).
You are working with data where negative values are significant and should be retained.

![image](https://github.com/user-attachments/assets/4b0e1a79-84eb-460c-9341-9f6ac0cda088)

### 3. ReLU Function

## Exponential Linear Units

### 1. Softmax Funciton

### 2. Softplus Function
