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

1. Non-linearity: Tanh introduces non-linearity to the model, which allows neural networks to learn complex patterns and relationships in the data. Without non-linear activation functions, a neural network would essentially behave as a linear model, no matter how many layers it has.

2. Centered Around Zero: The output of the tanh function is centered around 0, unlike the sigmoid function, which outputs values between 0 and 1. This makes the tanh activation function more useful for many types of tasks, as the mean of the output is closer to zero, leading to more efficient training and faster convergence.

3. Gradient Behavior: Tanh helps mitigate the vanishing gradient problem (to some extent), especially when compared to sigmoid activation. This is because the gradient of the tanh function is generally higher than that of the sigmoid, enabling better weight updates during backpropagation.

**Disadvantages of Tanh**

1. Vanishing Gradient Problem: Similar to the sigmoid function, tanh suffers from the vanishing gradient problem for large values of the input (both positive and negative). When the input to the tanh function is very large or very small, the gradient approaches zero, which can slow down or halt learning during backpropagation, especially in deep networks.

2. Not Suitable for All Tasks: While tanh works well in many cases, it might not be the best option for all types of neural network architectures. For instance, ReLU (Rectified Linear Unit) has gained popularity for deep networks due to its simplicity and efficiency in mitigating the vanishing gradient problem.

3. Sensitive to Outliers: Extreme values in the input can lead to saturated regions where the gradient is close to zero, making learning slow or ineffective. This could happen if the inputs to the tanh function are not scaled properly.

**When to Use Tanh**

1. The data is already normalized or centered around zero.

2. You are building shallow neural networks (i.e., networks with fewer layers).

3. You are working with data where negative values are significant and should be retained.

![image](https://github.com/user-attachments/assets/4b0e1a79-84eb-460c-9341-9f6ac0cda088)

### 3. ReLU Function

Rectified Linear Unit (ReLU) is a popular activation functions used in neural networks, especially in deep learning models. It has become the default choice in many architectures due to its simplicity and efficiency. The ReLU function is a piecewise linear function that outputs the input directly if it is positive; otherwise, it outputs zero.

In simpler terms, ReLU allows positive values to pass through unchanged while setting all negative values to zero. This helps the neural network maintain the necessary complexity to learn patterns while avoiding some of the pitfalls associated with other activation functions, like the vanishing gradient problem.

The ReLU function can be described mathematically as follows:
f(x)=max(0,x)

![image](https://github.com/user-attachments/assets/0c697e57-9d21-42ff-a93b-7a0a7feb6289)

**Why is ReLU Popular?**

1. Simplicity: ReLU is computationally efficient as it involves only a thresholding operation. This simplicity makes it easy to implement and compute, which is important when training deep neural networks with millions of parameters.

2. Non-Linearity: Although it seems like a piecewise linear function, ReLU is still a non-linear function. This allows the model to learn more complex data patterns and model intricate relationships between features.

3. Sparse Activation: ReLU's ability to output zero for negative inputs introduces sparsity in the network, meaning that only a fraction of neurons activate at any given time. This can lead to more efficient and faster computation.

4. Gradient Computation: ReLU offers computational advantages in terms of backpropagation, as its derivative is simple—either 0 (when the input is negative) or 1 (when the input is positive). This helps to avoid the vanishing gradient problem, which is a common issue with sigmoid or tanh activation functions.

**Drawbacks of ReLU**

1. Dying ReLU Problem: One of the most significant drawbacks of ReLU is the "dying ReLU" problem, where neurons can sometimes become inactive and only output 0. This happens when large negative inputs result in zero gradient, leading to neurons that never activate and cannot learn further.

2. Unbounded Output: Unlike other activation functions like sigmoid or tanh, the ReLU activation is unbounded on the positive side, which can sometimes result in exploding gradients when training deep networks.

3. Noisy Gradients: The gradient of ReLU can be unstable during training, especially when weights are not properly initialized. In some cases, this can slow down learning or lead to poor performance.

**When to Use ReLU?**

1. Handling Sparse Data: ReLU helps with sparse data by zeroing out negative values, promoting sparsity and reducing overfitting.

2. Faster Convergence: ReLU accelerates training by preventing saturation for positive inputs, enhancing gradient flow in deep networks.

**Variants Of ReLU**

1. Leaky ReLU

Leaky ReLU introduces a small slope for negative values instead of outputting zero, which helps keep neurons from "dying."

![image](https://github.com/user-attachments/assets/d59dcb42-fdff-49e1-9e1e-09bf6d2c154e)

where α is a small constant (often set to 0.01).

![image](https://github.com/user-attachments/assets/17cd329d-5993-4c1f-b0cc-d9647eba52cc)

2. Parametric ReLU

Parametric ReLU (PReLU) is an extension of Leaky ReLU, where the slope of the negative part is learned during training. The formula is as follows:

![image](https://github.com/user-attachments/assets/2e3303a0-5f00-4bea-8325-ca5efae754ea)

Where:
x is the input. α is the learned parameter that controls the slope for negative inputs. Unlike Leaky ReLU, where α is a fixed value (e.g., 0.01), PReLU learns the value of α\alphaα during training.

![image](https://github.com/user-attachments/assets/7af34f97-e653-411a-9146-9062d86eefa9)

In PReLU, α can adapt to different training conditions, making it more flexible compared to Leaky ReLU, where the slope is predefined. This allows the model to learn the best negative slope for each neuron during the training process.

## Exponential Linear Units

### 1. Softmax Funciton

Softmax function is a mathematical function that converts a vector of raw prediction scores (often called logits) from the neural network into probabilities. These probabilities are distributed across different classes such that their sum equals 1. Essentially, Softmax helps in transforming output values into a format that can be interpreted as probabilities, which makes it suitable for classification tasks.

In a multi-class classification neural network, the final layer outputs a set of values, each corresponding to a different class. These values, before Softmax is applied, can be any real numbers, and may not provide meaningful information directly. The Softmax function processes these values into probabilities, which indicate the likelihood of each class being the correct one.

Softmax gained prominence with the rise of deep learning, particularly in models such as multilayer perceptrons (MLPs) and convolutional neural networks (CNNs), where it is typically applied to the final output layer in classification tasks.

![image](https://github.com/user-attachments/assets/afd02543-0a47-4aa8-83d3-753b0f44beb2)

zi is the logit (the output of the previous layer in the network) for the i^{th}class.
K is the number of classes.

### 2. Softplus Function

