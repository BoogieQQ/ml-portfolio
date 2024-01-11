import numpy as np
import time
from scipy import special, sparse
from oracles import BinaryLogistic


class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций

        **kwargs - аргументы, необходимые для инициализации
        """

        l2_coef = kwargs['l2_coef'] if 'l2_coef' in kwargs.keys() else 1

        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(l2_coef)
        else:
            raise TypeError

        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.coefs_ = None

    def fit(self, X, y, w_0=None, b=0, trace=False, consider_tolerance=True, teach_bias=True):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        trace - переменная типа bool

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)

        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """

        if w_0 is None:
            n = X.shape[0]
            if n >= 1000:
                ind = np.random.choice(n, 100)
                w_0 = self._fit_w0(X[ind], y[ind], b)
            else:
                w_0 = np.zeros(X.shape[1])
                w_0 = np.insert(w_0, 0, b)
        else:
            w_0 = np.insert(w_0, 0, b)

        if isinstance(X, sparse.csr_matrix):
            X = sparse.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        else:
            X = sparse.csr_matrix(np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))

        history = None
        start_time = None
        if trace:
            history = {'time': [], 'func': [], 'weights': [], 'accuracy': []}

        cur_w = w_0
        Q_prev = 0
        Q_cur = self.oracle.func(X, y, cur_w)
        k = 0
        while (not consider_tolerance or abs(Q_cur - Q_prev) >= self.tol) and k <= self.max_iter:
            if trace:
                start_time = time.time()
            k += 1

            grad = self.oracle.grad(X, y, cur_w)
            if not teach_bias:
                grad[0] = 0
            lr = self.step_alpha / (k ** self.step_beta)
            cur_w = cur_w - lr * grad

            Q_prev = Q_cur
            Q_cur = self.oracle.func(X, y, cur_w)

            if trace:
                self.coefs_ = cur_w
                history['time'].append(time.time() - start_time)
                history['func'].append(Q_prev)
                history['weights'].append(cur_w[1:])
                y_pred = self.predict(X, True)
                history['accuracy'].append(np.sum(y_pred == y) / len(y))

        self.coefs_ = cur_w

        if trace:
            history['time'].append(time.time() - start_time)
            history['func'].append(Q_cur)
            history['weights'].append(cur_w[1:])
            y_pred = self.predict(X, True)
            history['accuracy'].append(np.sum(y_pred == y) / len(y))

        if trace:
            return history

    def predict(self, X, expanded=False):
        """
        Получение меток ответов на выборке X

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: одномерный numpy array с предсказаниями
        """
        return self.predict_proba(X, expanded).argmax(axis=1) * 2 - 1

    def predict_proba(self, X, expanded=False):
        """
        Получение вероятностей принадлежности X к классу k

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        return: двумерной numpy aяrray, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k
        """
        if not expanded and isinstance(X, sparse.csr_matrix):
            X = sparse.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        elif not expanded:
            X = sparse.csr_matrix(np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))
        logits = X @ self.coefs_

        probs = special.expit(logits)
        ones = np.ones(probs.shape)
        return np.hstack(((ones - probs).reshape(-1, 1), probs.reshape(-1, 1)))

    def get_objective(self, X, y, expanded=False):
        """
        Получение значения целевой функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: float
        """
        if not expanded and isinstance(X, sparse.csr_matrix):
            X = sparse.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        elif not expanded:
            X = sparse.csr_matrix(np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))
            
        return self.oracle.func(X, y, self.coefs_)

    def get_gradient(self, X, y, expanded=False):
        """
        Получение значения градиента функции на выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array

        return: numpy array, размерность зависит от задачи
        """
        if not expanded and isinstance(X, sparse.csr_matrix):
            X = sparse.hstack((np.ones(X.shape[0]).reshape(-1, 1), X))
        elif not expanded:
            X = sparse.csr_matrix(np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))
            
        return self.oracle.grad(X, y, self.coefs_)

    def get_weights(self):
        """
        Получение значения весов функционала
        """
        return self.coefs_

    def _fit_w0(self, X, y, b):
        self.fit(X, y, np.zeros(X.shape[1]), b)
        return self.coefs_


class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """

    def __init__(
            self, loss_function, batch_size, step_alpha=1, step_beta=0,
            tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs
    ):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора.
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия

        batch_size - размер подвыборки, по которой считается градиент

        step_alpha - float, параметр выбора шага из текста задания

        step_beta- float, параметр выбора шага из текста задания

        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход

        max_iter - максимальное число итераций (эпох)

        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.

        **kwargs - аргументы, необходимые для инициализации
        """

        l2_coef = kwargs['l2_coef'] if 'l2_coef' in kwargs.keys() else 1

        if loss_function == 'binary_logistic':
            self.oracle = BinaryLogistic(l2_coef)
        else:
            raise TypeError

        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tol = tolerance
        self.max_iter = max_iter
        self.coefs_ = None
        self.seed = random_seed

    def fit(self, X, y, w_0=None, b=0, trace=False, log_freq=1, consider_tolerance=True, teach_bias=True):
        """
        Обучение метода по выборке X с ответами y

        X - scipy.sparse.csr_matrix или двумерный numpy.array

        y - одномерный numpy array

        w_0 - начальное приближение в методе

        Если trace = True, то метод должен вернуть словарь history, содержащий информацию
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}

        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления.
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.

        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        if w_0 is None:
            n = X.shape[0]
            if n >= 1000:
                ind = np.random.choice(n, 100)
                w_0 = self._fit_w0(X[ind], y[ind], b)
            else:
                w_0 = np.zeros(X.shape[1])
                w_0 = np.insert(w_0, 0, b)
        else:
            w_0 = np.insert(w_0, 0, b)

        if isinstance(X, sparse.csr_matrix):
            X = sparse.csr_matrix(sparse.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))
        else:
            X = sparse.csr_matrix(np.hstack((np.ones(X.shape[0]).reshape(-1, 1), X)))

        history = None
        start_time = None
        if trace:
            history = {'time': [], 'func': [], 'epoch_num': [], 'weights_diff': [], 'weights': [], 'accuracy': []}

        cur_w = w_0
        Q_prev = 0
        Q_cur = self.oracle.func(X, y, cur_w)
        k = 0
        prev_pseudo_epoch = 0
        pseudo_epoch = 0
        if trace:
            start_time = time.time()

        while (not consider_tolerance or abs(Q_cur - Q_prev) >= self.tol) and k <= self.max_iter:
            k += 1
            n = X.shape[0]

            ind = np.arange(n)
            np.random.shuffle(ind)
            X_shuffled = X[ind]
            y_shuffled = y[ind]

            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                grad = self.oracle.grad(X_batch, y_batch, cur_w)

                lr = self.step_alpha / (k ** self.step_beta)
                if not teach_bias:
                    grad[0] = 0

                cur_w = cur_w - lr * grad
                Q_prev = Q_cur
                Q_cur = self.oracle.func(X, y, cur_w)

                pseudo_epoch += self.batch_size / n
                if trace and pseudo_epoch - prev_pseudo_epoch > log_freq:
                    self.coefs_ = cur_w
                    prev_pseudo_epoch = pseudo_epoch

                    history['epoch_num'].append(pseudo_epoch)
                    history['time'].append(time.time() - start_time)
                    history['func'].append(Q_cur)
                    history['weights_diff'].append(lr ** 2 * np.dot(grad, grad))
                    history['weights'].append(cur_w[1:])
                    y_pred = self.predict(X, True)
                    history['accuracy'].append(np.sum(y_pred == y) / len(y))
                    start_time = time.time()

        self.coefs_ = cur_w
        if trace:
            self.coefs_ = cur_w
            history['epoch_num'].append(pseudo_epoch)
            history['time'].append(time.time() - start_time)
            history['func'].append(Q_cur)
            history['weights_diff'].append(lr ** 2 * np.dot(grad, grad))
            history['weights'].append(cur_w[1:])
            y_pred = self.predict(X, True)
            history['accuracy'].append(np.sum(y_pred == y) / len(y))
            return history

    def _fit_w0(self, X, y, b):
        self.fit(X, y, np.zeros(X.shape[1]), b)
        return self.coefs_
