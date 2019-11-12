from types import SimpleNamespace

import numpy as np


def load_pos_data(file):
    """
    Returns a list of samples. Each sample has `text` and `labels` attributes.
    """

    data = [
        list(zip(*(
            line.split()
            for line
            in sample.split('\n')
        )))
        for sample
        in open(file).read().split('\n\n')
    ]

    return [
        SimpleNamespace(text=text, labels=labels)
        for text, labels
        in data
    ]


def load_vocabulary():
    words = set()
    for dataset in [load_pos_data('data/train'), load_pos_data('data/test')]:
        for sample in dataset:
            words = words.union(sample.text)

    vocabulary = {'<pad>': 0}  # Zero is reserved for padding in keras
    for i, word in enumerate(words):
        vocabulary[word] = i+1
    return vocabulary


pos_vocabulary = {
    'ADJ': 0,
    'ADP': 1,
    'ADV': 2,
    'AUX': 3,
    'CCONJ': 4,
    'DET': 5,
    'INTJ': 6,
    'NOUN': 7,
    'NUM': 8,
    'PART': 9,
    'PRON': 10,
    'PROPN': 11,
    'PUNCT': 12,
    'SCONJ': 13,
    'SYM': 14,
    'VERB': 15,
    'X': 16
}


def embedding_matrix(vocabulary):

    embeddings = {
        line.split()[0]: np.array(line.split()[1:])
        for line
        in open('data/embeddings')
    }

    matrix = np.zeros((
        len(vocabulary),
        len(list(embeddings.values())[0])
    ))

    for word, id in vocabulary.items():
        try:
            matrix[id] = embeddings[word]
        except KeyError:  # For <pad> token
            pass

    return matrix


def sine_dataset(num_samples, time_steps, skip):

    def sample(len_sample, skip):
        lin = np.arange(len_sample + skip) / 20.0
        a = np.random.rand() * 3 + 0.2
        b = np.random.randint(0, 100)
        c = np.random.rand() * 1.5
        d = (np.random.rand() - 0.5)
        data = np.sin(lin * a + b) * c + d
        return np.expand_dims(data[:len_sample], axis=-1), data[-1]

    data, targets = zip(*(sample(time_steps, skip) for _ in range(num_samples)))
    return np.array(data), np.array(targets)
