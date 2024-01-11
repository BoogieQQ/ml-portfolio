import numpy as np
from tqdm import tqdm


def grad_finite_diff(function, X, y, w, eps=1e-8):
    """
    Возвращает численное значение градиента, подсчитанное по следующией формуле:
        result_i := (f(w + eps * e_i) - f(w)) / eps,
        где e_i - следующий вектор:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    result = np.zeros(len(w))
    for i in tqdm(range(len(w))):
        e = np.zeros(len(w))
        e[i] = eps
        result[i] = (function(X, y, w + e) - function(X, y, w)) / eps
    return result
