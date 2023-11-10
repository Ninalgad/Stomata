from sklearn.model_selection import train_test_split
import numpy as np


def monotonic_incr(seq):
    for i in range(len(seq) - 1):
        seq[i + 1] = max(seq[i + 1], seq[i])
    return seq


def create_splits(x, y, d, f, seed):
    f_unique = sorted(set(f))
    f_train, f_test = train_test_split(f_unique, test_size=0.1, random_state=seed)
    f_train, f_test = set(f_train), set(f_test)
    train_idx = np.array([n in f_train for n in f])
    dev_idx = np.logical_not(train_idx)
    x_train, x_dev = x[train_idx], x[dev_idx]
    y_train, y_dev = y[train_idx], y[dev_idx]
    d_train, d_dev = d[train_idx], d[dev_idx]

    return x_train, x_dev, y_train, y_dev, d_train, d_dev
