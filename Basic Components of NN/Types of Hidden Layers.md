## Types of Hidden layers in ANNs

1. Dense layer - Most common type of hidden layer in an ANN. Every neuron in a dense layer is connected to every neuron in the previous and subsequent layers. This layer performs a weighted sum of inputs and applies an activation function to introduce non-linearity.

2. Convolutional error - Used in Convolutional Neural Networks (CNNs) for image processing tasks. They apply convolution operations to the input, capturing spatial hierarchies in the data. Convolutional layers use filters to scan across the input and generate feature maps. This helps in detecting edges, textures, and other visual features.

Role: Extracts spatial features from images.

Function: Applies convolution using filters.

![image](https://github.com/user-attachments/assets/2e2fdd40-cb92-4ae4-bed6-3be3748859f1)

3. Recurrent Layer - Used in RNNs for sequence data like time series or natural language. They have connections that loop back, allowing information to persist across time steps. This makes them suitable for tasks where context and temporal dependencies are important.

Role: Processes sequential data with temporal dependencies.
Function: Maintains state across time steps.

![image](https://github.com/user-attachments/assets/423e2a36-677e-4a43-928f-e7d2e5370356)

4. Dropout Layer - Regularization technique used to prevent overfitting. They randomly drop a fraction of the neurons during training, which forces the network to learn more robust features and reduces dependency on specific neurons. During training, each neuron is retained with a probability p.

Role: Prevents overfitting.

Function: Randomly drops neurons during training.

5. Pooling layer - Used to reduce the spatial dimensions of the data, thereby decreasing the computational load and controlling overfitting. Common types of pooling include Max Pooling and Average Pooling.

Use Cases: Dimensionality reduction in CNNs

![image](https://github.com/user-attachments/assets/8675e26b-7f69-4b94-bb48-7a16e3a1118b)

6. Batch Normalization Layer - 

normalizes the output of a previous activation layer by subtracting the batch mean and dividing by the batch standard deviation. This helps in accelerating the training process and improving the performance of the network.

Use Cases: Stabilizing and speeding up training

![image](https://github.com/user-attachments/assets/e530fe72-b1aa-4935-8612-9019a8d5a102)

