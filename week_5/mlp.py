import os

from week_4.backstage.utils import timestamp

import tensorflow as tf
import tensorflow.keras as keras
from tensorboard.plugins.hparams import api as hp


class MultilayerPerceptron(keras.Model):

    def __init__(self, dim_output, dim_hidden, activation, output_activation):
        super(MultilayerPerceptron, self).__init__(name='multilayer_perceptron')
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.layer_1 = keras.layers.Dense(
            units=dim_hidden,
            activation=activation)
        self.layer_2 = keras.layers.Dense(
            units=dim_output,
            activation=output_activation)

    def call(self, x):
        h = self.layer_1(x)
        y = self.layer_2(h)
        return y


def train(x, y, dim_output, **hparams):

    model = MultilayerPerceptron(
        dim_output=dim_output,
        dim_hidden=hparams['dim_hidden'],
        activation=hparams['activation'],
        output_activation=hparams['output_activation'])

    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=hparams['learning_rate']),
        loss=hparams['loss_function'],
        metrics=['accuracy'])

    hst = model.fit(
        x=x,
        y=y,
        batch_size=hparams['batch_size'],
        epochs=hparams['epoch'],
        validation_split=0.2,
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir=os.path.join("logs", timestamp()),
                histogram_freq=1)],
        verbose=0)

    best_validation_accuracy = max(hst.history['val_accuracy'])
    final_loss = hst.history["loss"][-1]

    # TensorBoard HParams saving
    with tf.summary.create_file_writer(os.path.join('logs', timestamp(), 'hparams')).as_default():
        hp.hparams(hparams)
        tf.summary.scalar('hparams_accuracy', best_validation_accuracy, step=0)

    return best_validation_accuracy, final_loss


