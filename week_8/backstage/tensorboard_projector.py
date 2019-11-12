"""
This code is heavily inspired by a hack posted in TensorBoard GitHub issues:

https://github.com/tensorflow/tensorboard/issues/2471
"""


import os

import tensorflow as tf
import numpy as np
from tensorboard.plugins import projector

TENSORBOARD_DIR = 'logs/embeddings/'
TENSORBOARD_METADATA_FILE = 'metadata'

embs = {
    line.split()[0]: np.array(line.split()[1:])
    for line
    in open('data/pos/embeddings')
}

with open(os.path.join(TENSORBOARD_DIR, TENSORBOARD_METADATA_FILE), 'w') as f:
    for word in embs.keys():
        f.write(word)
        f.write('\n')

vectors = np.array([vector for vector in embs.values()])
embeddings = tf.Variable(vectors, name='embeddings')
CHECKPOINT_FILE = TENSORBOARD_DIR + '/model.ckpt'
ckpt = tf.train.Checkpoint(embeddings=embeddings)
ckpt.save(CHECKPOINT_FILE)

reader = tf.train.load_checkpoint(TENSORBOARD_DIR)
map = reader.get_variable_to_shape_map()
key_to_use = ""
for key in map:
    if "embeddings" in key:
        key_to_use = key

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = key_to_use
embedding.metadata_path = TENSORBOARD_METADATA_FILE

writer = tf.summary.create_file_writer(TENSORBOARD_DIR)
projector.visualize_embeddings(writer, config, TENSORBOARD_DIR)  # As of 2.0.0 we need to add logdir into visualize_embeddings method manually
