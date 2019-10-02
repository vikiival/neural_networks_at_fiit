import numpy as np

from model import LinearRegressionModel

learning_rate = 0.03
num_epochs = 10
batch_size = 1

m = LinearRegressionModel(
        input_dim=1,
        learning_rate=learning_rate,
        w=[2],
        b=-2)


d = (
        np.random.rand(9, 3),
        np.array([i for i in range(9)])
    )


def test_predict():
        for predicted, expected in zip(
            m.predict(d[0]),
            [0.8, 0.95, 0.85, 1.05]
        ):
            print(predicted, expected, round(predicted, 2) == expected)


def test_gradient():
        dw, db = m.gradient(*d)
        print(len(dw) == 1, len(dw))
        print(dw[0] == -1.3375)
        print(db == -0.625)

test_gradient()