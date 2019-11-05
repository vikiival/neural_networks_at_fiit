import itertools
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def conv_plot(kernel_size=3, stride=1, padding=True, speed=1):

    assert(kernel_size in [1, 3, 5])
    assert(stride in [1, 2, 3])
    assert(padding in [True, False])
    assert(padding or stride == 1)  # Without padding set stride to 1

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    image = np.random.rand(7, 7)

    # Background
    for ax_id, ax in enumerate(axes):
        ax.axis([0, 11, 0, 11])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        for spine in ['bottom', 'top', 'left', 'right']:
            ax.spines[spine].set_visible(False)

    axes[0].set_title('Input $X$')
    for i in range(7):
        for j in range(7):
            axes[0].add_patch(
                Rectangle(
                    xy=(i + 2, j + 2),
                    width=1,
                    height=1,
                    edgecolor='black',
                    facecolor=str(image[i, j])
                ))

    output_size = {
        1: 7,
        2: 4,
        3: 3
    }[stride]
    if not padding:
        output_size -= (kernel_size // 2) * 2
    shift = 2 + (7 - output_size) // 2

    axes[1].set_title('Output $Z$')
    for i in range(output_size):
        for j in range(output_size):
            axes[1].add_patch(
                Rectangle(
                    xy=(i + shift, 10 - j - shift),
                    width=1,
                    height=1,
                    edgecolor='black',
                    facecolor='1.0'
                ))

    for j, i in itertools.product(range(output_size), range(output_size)):
        output_coord = i + shift, 10 - j - shift
        res = Rectangle(output_coord, 1, 1, edgecolor='black', facecolor=str(np.random.rand()))
        rec_1 = Rectangle(output_coord, 1, 1, edgecolor='red', facecolor='none', linewidth='3')
        axes[1].add_patch(res)
        axes[1].add_patch(rec_1)

        x, y = i * stride + 2, 8 - j * stride
        x -= kernel_size // 2
        y -= kernel_size // 2
        if not padding:
            x += kernel_size // 2
            y -= kernel_size // 2
        rec_0 = Rectangle(
            (x, y),
            kernel_size, kernel_size,
            edgecolor='red', facecolor='none', linewidth='3')
        axes[0].add_patch(rec_0)

        fig.canvas.draw()
        sleep(0.5 / speed)
        rec_0.remove()
        rec_1.remove()
