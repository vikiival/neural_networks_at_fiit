import matplotlib.pyplot as plt


def show_sines(data, targets, skip):

    fig, axes = plt.subplots(1, 4, figsize=(9, 3))
    plt.subplots_adjust(0.05, 0.1, 0.95, 0.9, 0.3, 0.1)

    for ax, x, y in zip(axes, data, targets):
        ax.plot(x)
        ax.scatter(len(x) + skip, y)
        ax.set_ylim(-2.1, 2.1)

    fig.show()
