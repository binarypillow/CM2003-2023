import numpy as np


def k_fold(data, labels, n_folds, n_iter):
    # split data and labels in n folds
    indices = np.arange(data.shape[0])
    ind_split = np.array_split(indices, n_folds)

    # select fold for validation
    valid_indices = ind_split[n_iter]
    x_valid, y_valid = data[valid_indices], labels[valid_indices]

    # select folds for training
    train_indices = np.concatenate(
        [ind_split[i] for i in range(n_folds) if i != n_iter]
    )
    x_train, y_train = data[train_indices], labels[train_indices]

    return x_train, y_train, x_valid, y_valid
