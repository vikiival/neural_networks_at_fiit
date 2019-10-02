import numpy as np


class LinearRegressionModel:
    """
    Class for Linear Regression model

    Attributes
    ----------
    w : np.array dim=(input_dim)
        An array of weights.
    b : float
        Bias parameter.
    input_dim : int
        Number of features in samples
    lr : float
        Learning rate for gradient descent algorithm
    """

    def __init__(self, input_dim, learning_rate=0.03, w=None, b=.0):
        self.input_dim = input_dim
        self.lr = learning_rate
        self.w = np.zeros(input_dim) if w is None else w
        self.b = b

    def linear_regression(self, val):
        return np.sum(np.multiply(val, self.w)) + self.b

    def predict(self, xs):
        """
        Runs the linear regression model over multiple input vectors xs. xs is a matrix, where i-th row is a feature
        vector of i-th sample. Method returns a numpy array where i-th element is a prediction for i-th sample from xs.

        :param xs: 2D np.array dim=(num_samples, input_dim)
        :return: np.array dim=(num_samples)
        """
        
        return np.array(list(map(self.linear_regression, xs)))

    def biasDerivation(self, xs, ys):
        yp = self.predict(xs)
        length = len(ys)
        return (-2 / length) * (np.sum(ys - yp))

    def weightDerivation(self, xs, ys):
        yp = self.predict(xs)
        length = len(ys)
        # return np.sum((ys - yp) * xs.transpose(), axis = 0) #transpose a axis = 1
        return (-2 / length) * (ys - yp) @ xs


    def gradient(self, xs, ys):
        """
        Computes the derivatives of loss function L w.r.t parameters w and b. ys is a vector, where i-th element
        is a true score for i-th sample.

        xs and ys will have the same interpretation for following methods as well.

        :param xs:  2D np.array dim=(num_samples, input_dim)
        :param ys:  np.array dim=(num_samples)
        :return:    np.array dim=(input_dim), float
        """

        dw = self.weightDerivation(xs, ys)
        db = self.biasDerivation(xs, ys)
        return dw, db

    def gradient_descent(self, xs, ys, num_steps=100):
        """
        Performs the gradient descent algorithm for num_steps steps. Returns the final parameters and
        loss function value.

        :param      num_steps: int
        :return:    np.array dim=(input_dim), float, float
        """
        for _ in range(num_steps):
            self.step(xs, ys)


    def step(self, xs, ys):
        """
        Performs one gradient descent step and updates the parameters accordingly.
        """
        dw, db = self.gradient(xs, ys)
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db

    def xyz(self, predictedValues, realValues):
        return (predictedValues - realValues) ** 2

    def loss(self, xs, ys):
        """
        Calculates the loss L with current parameters for given data.

        :return: float
        """
        yp = self.predict(xs)
        length = len(xs)

        return np.sum(self.xyz(yp, ys)) / length

    def stochastic_gradient_descent(self, xs, ys, num_epochs=10, batch_size=2):
    """
    Performs a stochastic gradient descent training over the data xs, ys.

    :param num_epochs: Number of epochs for the training.
    :param batch_size: Number of samples in batches.
    :return: None
    """
    for _ in range(num_epochs):
        ...
        for _ in range(...):
            self.step(..., ...)
    # FIXME: 2.13
