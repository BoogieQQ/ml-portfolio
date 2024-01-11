import numpy as np
from scipy import special, sparse


class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """

    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')


class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.

    Оракул должен поддерживать l2 регуляризацию.
    """

    def __init__(self, l2_coef):
        """
        Задание параметров оракула.

        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """
        pred = X @ w 
        zero = np.zeros(len(pred))
        return (np.mean(np.logaddexp(zero, -y * pred))
                + self.l2_coef / 2 * np.dot(w[1:], w[1:]))

    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w - одномерный numpy array
        """

        if not isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(X)

        pred = X @ w
        return (-np.mean((X.T.multiply(y * special.expit(y*pred)*np.exp(np.clip((-1)*y*pred, -10e20, 700)))).T, axis=0) 
                +self.l2_coef*np.r_[0, w[1:]]).A[0]
