import types

import matplotlib.pyplot as plt


def add_magic(model, test_data=None, acc=False):
    model._m = types.SimpleNamespace()

    def wrap_sgd(model_sgd):
        def new_sgd(self, xs, ys, **kwargs):
            self._m.fig = plt.figure()
            self._m.history = dict(zip(
                ['train_loss', 'test_loss', 'train_acc', 'test_acc'],
                [[] for _ in range(4)]
            ))
            if test_data:
                self._m.test_data = test_data
            model_sgd(xs, ys, **kwargs)
        return new_sgd

    def wrap_epoch(model_epoch):
        def new_epoch(self, xs, ys, learning_rate):
            plt.clf()
            plt.grid()
            model_epoch(xs, ys, learning_rate)
            self._m.history['train_loss'].append(self.loss(xs, ys))
            plt.plot(self._m.history['train_loss'], 'r-', label='train loss')
            if acc:
                self._m.history['train_acc'].append(self.accuracy(xs, ys))
                plt.plot(self._m.history['train_acc'], 'r--', label='train acc')
            if test_data:
                self._m.history['test_loss'].append(self.loss(*self._m.test_data))
                plt.plot(self._m.history['test_loss'], 'g-', label='test loss')
                if acc:
                    self._m.history['test_acc'].append(self.accuracy(*self._m.test_data))
                    plt.plot(self._m.history['test_acc'], 'g--', label='test acc')
            plt.legend(loc='upper left')

            self._m.fig.canvas.draw()
        return new_epoch

    model.epoch = types.MethodType(wrap_epoch(model.epoch), model)
    model.stochastic_gradient_descent = types.MethodType(wrap_sgd(model.stochastic_gradient_descent), model)