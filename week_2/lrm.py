import numpy as np

from model import LinearRegressionModel

m = LinearRegressionModel(
            input_dim=1,
            w=np.array([0.5]),
            b=0.2)


d = (
            np.array([
                [1.2],
                [1.5],
                [1.3],
                [1.7]
            ]),
            np.array([0.2, 0.4, 0.5, 3.8])
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