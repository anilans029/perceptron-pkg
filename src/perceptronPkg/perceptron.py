import numpy as np
import os
import joblib
import logging as log


class Perceptron:
    def __init__(self, eta: float = None, epochs: int = None):
        self.weights = np.random.randn(3) * 1e-4  ### small random weights
        training = (eta is not None) and (epochs is not None)
        if training:
            log.info(f"initial weights before training :  \n{self.weights}")
        self.eta = eta
        self.epochs = epochs

    def _z_outcome(self, inputs, weights):
        """
        It takes in the inputs and weights, and returns the dot product plus the bias

        :param inputs: a list of the inputs to the neuron
        :param weights: a list of weights for each of the inputs
        """

        return np.dot(inputs, weights)

    def activation_function(self, z):
        """
        The activation function takes in a number and returns 0 if the number is negative, and 1 if the number is positive

        :param z: the weighted sum of the inputs
        """
        return np.where(z > 0, 1, 0)

    def fit(self, X, y):
        """
        The function takes in two arguments, X and y, and fit or train a model

        :param X: The input data
        :param y: The target variable
        """
        self.X = X
        self.y = y

        X_with_bias = np.c_[self.X, -np.ones((len(self.X), 1))]
        log.info(f"X with bias :\n {X_with_bias}")

        for epoch in range(self.epochs):
            log.info("--" * 13)
            log.info(f"for epoch >> {epoch + 1}")
            log.info("--" * 13)

            z = self._z_outcome(X_with_bias, self.weights)
            y_hat = self.activation_function(z)
            log.info(f"predicted value after forward pass : \n{y_hat}")

            self.error = self.y - y_hat
            log.info(f"error : \n{self.error}")

            self.weights = self.weights + self.eta * np.dot(X_with_bias.T, self.error)
            log.info(f"updated weights after epoch : {epoch + 1}/{self.epochs}: \n {self.weights}")
            log.info("##" * 13)

    def prediction(self, test_inputs):
        """
        The function takes in a list of inputs, and returns a list of predictions.

        :param test_inputs: The inputs to the neural network, the same way as training
        """
        X_with_bias = np.c_[test_inputs, -np.ones((len(test_inputs), 1))]
        z = self._z_outcome(X_with_bias, self.weights)
        return self.activation_function(z)

    def total_loss(self):
        total_loss = np.sum(self.error)
        log.info(f"total loss : {total_loss}\n")

    def _create_dir_return_path(self, model_dir, filename):
        """
        It creates a directory if it doesn't exist, and returns the path to the file

        :param model_dir: The directory where the model will be saved
        :param filename: the name of the file to be saved
        """
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)

    def save(self, filenamem, model_dir=None):
        """
        The function saves the model to a file  in the model_dir directory

        :param filenamem: The name of the file to save the model to
        :param model_dir: The directory where the model will be saved. If not specified, the model will be saved in a
        temporary directory
        """
        if model_dir is not None:
            model_file_path = self._create_dir_return_path(model_dir, filenamem)
            joblib.dump(self, model_file_path)

        else:
            model_file_path = self._create_dir_return_path("model", filenamem)
            joblib.dump(self, model_file_path)

    def load(self, filepath):
        """
        It loads the filepath.

        :param filepath: The path to the file to load
        """
        return joblib.load(filepath)