# NN_numbers
MNIST Digit Classifier with HTTP Server for Predictions

This project demonstrates a TensorFlow-based neural network for classifying handwritten digits from the MNIST dataset. It includes an HTTP server that accepts POST requests with pixel data, processes the data, and returns the predicted digit with the associated accuracy.

Features:

MNIST Dataset Integration: Utilizes the MNIST dataset for training and testing the neural network model. The dataset is normalized to enhance training efficiency.

Neural Network Architecture: A simple sequential neural network with two hidden layers, each consisting of 50 neurons and ReLU activation functions. The output layer uses a softmax activation function for digit classification.

Training and Evaluation: The model is trained using the Adam optimizer and sparse categorical cross-entropy loss function. Training is performed in batches of 32 images over 5 epochs.

HTTP Server: Implements an HTTP server that listens for POST requests containing pixel data. The server preprocesses the data to match the MNIST format, predicts the digit, and returns the result along with the prediction accuracy.

Key Components:

normalize(images, labels): Function to normalize image pixel values from the range [0, 255] to [0, 1].

model: Defines the neural network architecture and compiles it with the specified optimizer, loss function, and metrics.

train_dataset and test_dataset: Preprocessed and batched training and testing datasets.

SimpleHTTPRequestHandler: A custom HTTP request handler class that processes POST requests, performs predictions, and sends back the response.

Usage:

Data Loading and Normalization: Load the MNIST dataset and normalize the images using the normalize function.
Model Definition and Compilation: Define the neural network architecture and compile it.
Training: Train the model with the training dataset.
HTTP Server Setup: Start the HTTP server to listen for incoming POST requests.
Prediction Requests: Send POST requests to the server with pixel data of handwritten digits to receive predictions.
Dependencies:

Python 3.x
TensorFlow
TensorFlow Datasets
NumPy
Matplotlib (for optional visualization)
Logging
Example POST Request Data Format:

The server expects a POST request with the following data format:

javascript
Copiar código
pixeles=<comma-separated list of 784 pixel values>
The pixel values should be normalized between 0 and 1.

Starting the Server:

To start the server, run the script and it will listen on localhost at port 8000:

python
Copiar código
print("Iniciando el servidor...")
server = HTTPServer(('localhost', 8000), SimpleHTTPRequestHandler)
server.serve_forever()
This project provides an end-to-end solution for training a digit classifier and serving predictions via an HTTP server, making it a versatile tool for various applications requiring digit recognition.
