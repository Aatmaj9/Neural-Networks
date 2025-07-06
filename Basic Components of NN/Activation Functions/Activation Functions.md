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

**1. Leaky ReLU**

Leaky ReLU introduces a small slope for negative values instead of outputting zero, which helps keep neurons from "dying."

![image](https://github.com/user-attachments/assets/d59dcb42-fdff-49e1-9e1e-09bf6d2c154e)

where α is a small constant (often set to 0.01).

![image](https://github.com/user-attachments/assets/17cd329d-5993-4c1f-b0cc-d9647eba52cc)

**2. Parametric ReLU**

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

zi is the logit (the output of the previous layer in the network) for the i th class.
K is the number of classes.

**Working of softmax**

**Step 1: Raw Logits (Pre-Softmax Outputs)**
Consider the output from the last layer of the neural network, which consists of logits. These logits are unbounded real numbers and represent the raw predictions for each class.

Let’s assume we are working on a classification task with K classes. The neural network provides an output vector z = {z1, z2, ......... zK} where each z(i) is the logit corresponding to the i th class.

**Step 2: Applying the Softmax Function**
The Softmax function transforms these logits into probabilities. The formula for Softmax for each class i is:

![image](https://github.com/user-attachments/assets/36788cf1-d6ae-4f85-a1ad-49974b6a3d5b)

**Step 3: Exponential Scaling**
The exponential function e^(z(i)) applied to each logit zi plays a crucial role. It emphasizes the difference between logits: even a slight increase in a logit value leads to a larger probability, while small logits result in near-zero probabilities.

**Step 4: Normalization**
The sum of the exponentials is used to normalize the values into probabilities. The normalization step ensures that all the probabilities add up to 1.

**Step 5: Interpretation of the Output**
The result of applying the Softmax function to the logits is a probability distribution. Each element represents the probability that the input data belongs to a particular class.

**Advantages of the Softmax Function**

1. Probability Distribution: Softmax produces output values that can be interpreted as probabilities that provides a clear way to measure the confidence of a model in its predictions.

2. Differentiability: Softmax function is differentiable, hence, we can integrate into the backpropagation algorithm used to train neural networks. This ensures that model parameters can be updated effectively using gradient descent.

3. Normalized Output: Softmax ensures that the sum of the output probabilities equals 1. This allows the results to be interpreted in a probabilistic manner, where the class with the highest probability is the model's prediction.

**Drawbacks of the Softmax Function**

1. Large disparities in logits can dominate the output, making Softmax sensitive to outliers and noisy data.

2. Small probabilities can cause very small gradients during backpropagation, slowing down learning.

3. Softmax may assign high probabilities to incorrect classes, leading to overly confident predictions.

4. Requires exponentiation and normalization, making it computationally expensive for large datasets or many classes.

5. Not suited for multi-label tasks, where an instance can belong to multiple classes.

[Softmax Implementation](Neural-Netwroks/Basic Components of NN/Activation Functions/Softmax/Softmax.md)

### 2. Softplus Function

Softplus function is a smooth approximation of the ReLU function, defined mathematically as:

![image](https://github.com/user-attachments/assets/066633f0-7b57-4ebb-8e2c-dc5c25fc263b)

![image](https://github.com/user-attachments/assets/8c4743e2-d8f6-4d33-afbc-d6e7322ce656)

**Characterisitics**

1. The output is always positive.

2. The function smoothly increases and is always differentiable, unlike ReLU, which has a sharp corner at zero.

3. For negative inputs, the function approaches zero, but unlike ReLU, it never exactly reaches zero, avoiding the problem of "dying neurons."

**Mathematical properties**

The Softplus function has some important mathematical properties that are helpful for neural networks:

1. Derivative of Softplus: The derivative of the Softplus function is the sigmoid function. This property makes Softplus useful in situations where we want to control the smoothness of the gradient, as it has a continuous and smooth derivative.

![image](https://github.com/user-attachments/assets/8375c806-28bd-4aa2-ad44-6ea5a8b02669)

![image](https://github.com/user-attachments/assets/7742345d-fd30-4b02-9aed-b31e6ea64b68)

2. Range: The Softplus function outputs values from 0 to infinity. This ensures that it can be used in situations where positive outputs are desired, such as in regression tasks where the outputs should be non-negative.

3. Behavior at Extremes:

As x→∞, Softplus behaves like a linear function: 

lim  x→ ∞ ln(1+e^(x))≈ x

As x → −∞, Softplus approaches zero, but never actually reaches zero. This helps to avoid the problem of dead neurons, which is common in ReLU when the input is negative:

lim x → −∞ ln(1+e^(x))≈0

**Advantages of Softplus**

1. Smooth Non-linearity: The smoothness of the Softplus function makes it a good choice for models where smooth and continuous transitions are important, such as in certain types of regression and classification problems.

2. Solves the Dying Neuron Problem: Softplus avoids the "dying neuron" problem of ReLU by allowing negative inputs to produce very small but non-zero outputs, ensuring that the neurons remain active during training.

3. Differentiable Everywhere: Unlike ReLU, which has a sharp corner at zero, Softplus is differentiable everywhere. This makes it more suitable for optimization algorithms that rely on gradients, as the gradients will be smooth and continuous.

4. Better Handling of Negative Inputs: Softplus handles negative inputs more gracefully than ReLU. While ReLU simply outputs zero for negative inputs, Softplus produces a small positive output, making it more appropriate for networks that need to work with both positive and negative data.

**Disadvantages of Softplus**

1. Computationally More Expensive: While Softplus is smooth and differentiable, it is computationally more expensive than ReLU because it requires computing the logarithm and exponential functions. This can slow down training for large networks, especially on resource-constrained systems.

2. Not as Popular as ReLU: While Softplus offers advantages, it is not as widely used as ReLU. ReLU has become the default choice for many architectures because it is computationally simpler and performs well in practice.

3. Slower Convergence: The smoother nature of Softplus can sometimes lead to slower convergence during training compared to ReLU, which may be a trade-off in certain applications.

**Softplus is useful when:**

1. You need a smooth and continuous activation function.

2. You want to avoid the dying neuron problem that occurs with ReLU.

3. The network deals with both positive and negative inputs, and you want the output to remain non-negative.

4. You prefer a differentiable function throughout the network for smoother gradient-based optimization.

However, for many deep learning models, ReLU or Leaky ReLU might still be preferred due to their simpler computation and better convergence in certain contexts.
