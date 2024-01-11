import numpy as np


def euclidean_distance(X: np.array, Y: np.array) -> np.array:
    n, m = X.shape[0], Y.shape[0]

    Z = np.dot(X, Y.T)
    X = np.einsum('ij,ij->i', X, X).reshape(n, 1)
    Y = np.einsum('ij,ij->i', Y, Y)

    X = np.repeat(X, repeats=m, axis=1)
    Y = np.tile(Y, (n, 1))

    return (X-2*Z+Y)**0.5


def cosine_distance(X: np.array, Y: np.array) -> np.array:
    Z = X @ Y.T
    n, m = Z.shape[0], Z.shape[1]

    X = np.linalg.norm(X, axis=1)
    X = np.repeat(X.reshape(n, 1), m, axis=1)

    Y = np.linalg.norm(Y, axis=1)
    Y = np.tile(Y, (n, 1))

    return 1 - Z/(X*Y)
