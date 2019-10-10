import numpy as np


class MultilayerPerceptron:

    def __init__(self, dim_input, dim_hidden, dim_output):

        self.w_1 = self.init_weights(dim_hidden, dim_input)
        self.b_1 = self.init_biases(dim_hidden)

        self.w_2 = self.init_weights(dim_output, dim_hidden)
        self.b_2 = self.init_biases(dim_output)

    def init_weights(self, dim_out, dim_in):
        """
        Xavier initialization for weights matrices. dim_out is number of rows, dim_in is number of columns.

        :param dim_out: int
        :param dim_in: int
        :return: np.array dim=(dim_out, dim_in)
        """
        weights = np.random.rand(dim_out, dim_in) * 2 - 1
        scale = np.math.sqrt(6 / (dim_out + dim_in))
        return weights * scale

    def init_biases(self, dim):
        """
        Zero initialization for biases. dim is layer size.

        :param dim: int
        :return: np.array dim=(dim)
        """
        return np.zeros(dim)

    def loss(self, xs, ys):
        return np.mean([
            (y - self.predict(x)[0])**2
            for x, y
            in zip(xs, ys)
        ])

    def stochastic_gradient_descent(self, xs, ys, num_epochs=500, learning_rate=0.003):
        for _ in range(num_epochs):
            self.epoch(xs, ys, learning_rate)

    def epoch(self, xs, ys, learning_rate):
        order = np.random.permutation(len(ys))
        xs, ys = xs[order], ys[order]
        for x, y in zip(xs, ys):
            self.step(x, y, learning_rate)

    def step(self, x, y, learning_rate):
        params = [self.w_1, self.b_1, self.w_2, self.b_2]
        dparams = self.gradient(x, y)
        for param, dparam in zip(params, dparams):
            param -= learning_rate * dparam

    def sigma(self, x):
        """
        Logistic activation function

        :param x: np.array
        :return: np.array dim=(dim_x)
        """
        return 1 / (1 + np.exp(-x))

    def dsigma(self, x):
        """
        Derivation of logistic activation function

        :param x: np.array
        :return: np.array dim=(dim_x)
        """
        sigma = self.sigma(x)
        return sigma * (1 - sigma)

    def predict(self, x):
        """
        Returns the prediction. It also returns some quantities that were calculated during the forwards pass.

        :param x: np.array dim=(dim_input)
        :return:
            y_hat: np.array dim=?
            z_2:   np.array dim=?
            h:     np.array dim=?
            z_1:   np.array dim=?
        """

        z_1 = self.w_1 @ x + self.b_1
        h = self.sigma(z_1)
        z_2 = self.w_2 @ h + self.b_2
        y_hat = self.sigma(z_2)
        return y_hat, z_2, h, z_1

    def xyz(self, y, yp, z_2):
        a = 2 * (yp - y)
        b = self.dsigma(z_2)
        return np.multiply(a, b)

    def gradient(self, x, y):
        """
        Returns the derivatives for parameters. First it calculates the forwards pass with self.predict() and then
        it does the backward pass.

        :param x: np.array dim=(dim_input)
        :param y: np.array dim=(dim_output)
        :return:
            dw_1: np.array dim=?neuro
            db_1: np.array dim=?
            dw_2: np.array dim=?
            db_2: np.array dim=?
        """
        y_hat, z_2, h, z_1 = self.predict(x)
        dz_2 = self.xyz(y, y_hat, z_2)  # FIXME: 3.4.2
        dw_2 = dz_2.reshape(len(dz_2), 1) @ h.reshape(len(h), 1).transpose()
        db_2 = dz_2

        dz_1 = (self.w_2.transpose() @ dz_2) * self.dsigma(z_1)
        dw_1 = dz_1.reshape(len(dz_1), 1) @ x.reshape(len(x), 1).transpose()
        db_1 = dz_1

        ...  # Continue with first layer parameters
        return dw_1, db_1, dw_2, db_2

    def accuracy(self, xs, ys):
        """
        Measures the accuracy of the model, i.e. how many xs has the right y predicted.

        :return: float
        """
        correctlyPredicted = 0
        for index, row in enumerate(xs):
            y_hat, z_2, h, z_1 = self.predict(row)
            indexOfMax = np.argmax(y_hat)
            if ys[index][indexOfMax]:
                correctlyPredicted += 1
            
        # for each row in xs make predict
        # take only y_hat
        # y_hat get index of biggest
        # ys should have one on same index
        # if yes counter++
        # return counter / len xs
        return correctlyPredicted / len(xs)
# hint: transponse has no effect on vector
