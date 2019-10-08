import types

import numpy as np


def load_data(filename, num_classes, normalize=False, *args):
    data = np.genfromtxt(f'data/{filename}', *args)
    np.random.seed(0)
    order = np.random.permutation(data.shape[0])
    data = data[order]
    output = types.SimpleNamespace()
    output.x = data[:, :-1]
    if normalize:
        means, stds = np.mean(output.x, axis=0), np.std(output.x, axis=0)
        output.x = (output.x - means) / stds
    num_samples = data.shape[0]
    output.y = np.zeros((num_samples, num_classes))
    output.y[
        np.arange(num_samples),
        data[:, -1].astype(int)
    ] = 1
    return output
