# Neural-Networks
Neural networks are machine learning models that mimic the complex functions of the human brain. These models consist of interconnected nodes or neurons that process data, learn patterns and enable tasks such as pattern recognition and decision-making.

![image](https://github.com/user-attachments/assets/bb83e25a-f724-4034-83b4-883712a2bae0)
### These networks are built from several key components:

Neurons: The basic units that receive inputs, each neuron is governed by a threshold and an activation function.

Connections: Links between neurons that carry information, regulated by weights and biases.

Weights and Biases: These parameters determine the strength and influence of connections.

Propagation Functions: Mechanisms that help process and transfer data across layers of neurons.

Learning Rule: The method that adjusts weights and biases over time to improve accuracy.

### Layers in Neural Network Architecture

Input Layer: This is where the network receives its input data. Each input neuron in the layer corresponds to a feature in the input data.

Hidden Layers: These layers perform most of the computational heavy lifting. A neural network can have one or multiple hidden layers. Each layer consists of units (neurons) that transform the inputs into something that the output layer can use.

Output Layer: The final layer produces the output of the model. The format of these outputs varies depending on the specific task like classification, regression.

![image](https://github.com/user-attachments/assets/5a25e7a6-6727-4c5b-a159-c2a9b05c7e8f)

## Working of Neural Networks

1. Forward Propagation

When data is input into the network, it passes through the network in the forward direction, from the input layer through the hidden layers to the output layer. This process is known as forward propagation. Here’s what happens during this phase:

a) Linear Transformation: Each neuron in a layer receives inputs which are multiplied by the weights associated with the connections (weighted sum). These products are summed together and a bias is added to the sum. 

b) Activation: The result of the linear transformation (denoted as z) is then passed through an activation function. The activation function is crucial because it introduces non-linearity into the system, enabling the network to learn more complex patterns. Popular activation functions include ReLU, sigmoid and tanh.

c) Propagation - The output of one layer becomes the input for the next layer and the process repeats until the final layer produces the network's prediction.

2. Back Propagation 

After forward propagation, the network evaluates its performance using a loss function which measures the difference between the actual output and the predicted output. The goal of training is to minimize this loss. This is where backpropagation comes into play:

a) Loss Calculation: The network calculates the loss which provides a measure of error in the predictions. The loss function could vary; common choices are mean squared error for regression tasks or cross-entropy loss for classification.
b) Gradient Calculation: The network computes the gradients of the loss function with respect to each weight and bias in the network. This involves applying the chain rule of calculus to find out how much each part of the output error can be attributed to each weight and bias.
c) Weight Update: Once the gradients are calculated, the weights and biases are updated using an optimization algorithms. The weights are adjusted in the opposite direction of the gradient to minimize the loss. The size of the step taken in each update is determined by the learning rate.

3. Iteration

This process of forward propagation, loss calculation, backpropagation and weight update is repeated for many iterations over the dataset. Over time, this iterative process reduces the loss and the network's predictions become more accurate.Through these steps, neural networks can adapt their parameters to better approximate the relationships in the data, thereby improving their performance on tasks such as classification, regression or any other predictive modeling.

***
> **_NOTE:_** Weights and Biases

A bias is a constant added to a neuron's weighted input. It is not linked to any specific input but shifts the activation function to fit the data. Biases enhance the flexibility and learning capacity of neural networks. While weights control the influence of inputs, biases act as offsets that allow neurons to activate under a wider range of conditions.

Purpose: Biases allow neurons to learn even when the weighted sum of inputs is insufficient, providing a mechanism to recognize patterns that don't pass through the origin.

Functionality: If biases are not present, neurons can only activate when the input reaches a specific threshold. Activation becomes more flexible when biases are present.

Training: Both weights, biases are updated during backpropagation to minimize prediction error. They help fine-tune neuron outputs, contributing to more accurate model performance.

***

## Learning of a Neural Network

1. Learning with Supervised Learning
In supervised learning, a neural network learns from labeled input-output pairs provided by a teacher. The network generates outputs based on inputs and by comparing these outputs to the known desired outputs, an error signal is created. The network iteratively adjusts its parameters to minimize errors until it reaches an acceptable performance level.

2. Learning with Unsupervised Learning
Unsupervised learning involves data without labeled output variables. The primary goal is to understand the underlying structure of the input data (X). Unlike supervised learning, there is no instructor to guide the process. Instead, the focus is on modeling data patterns and relationships, with techniques like clustering and association commonly used.

3. Learning with Reinforcement Learning
Reinforcement learning enables a neural network to learn through interaction with its environment. The network receives feedback in the form of rewards or penalties, guiding it to find an optimal policy or strategy that maximizes cumulative rewards over time. This approach is widely used in applications like gaming and decision-making
