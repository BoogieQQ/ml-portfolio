import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import time


class RandomForestMSE:
    def __init__(
            self, n_estimators, max_depth=None, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size \
            if feature_subsample_size is not None else 1. / 3
        self.params = trees_parameters

    def fit(self, X, y, X_val=None, y_val=None, logging=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects

        X_val : numpy ndarray
            Array of size n_val_objects, n_features

        y_val : numpy ndarray
            Array of size n_val_objects
        """
        self.trees = []
        self.indeces_for_trees = []
        n, m = X.shape
        if logging:
            logs = {'train_loss': [], 'train_metric': [], 'valid_loss': [], 'valid_metric': [], 'time': time.time()}
        for _ in range(self.n_estimators):

            if logging:
                y_pred = self.predict(X)
                logs['train_loss'].append(mean_squared_error(y, y_pred, squared=False))
                logs['train_metric'].append(mean_absolute_percentage_error(y, y_pred))
                if X_val is not None and y_val is not None:
                    y_pred = self.predict(X_val)
                    logs['valid_loss'].append(mean_squared_error(y_val, y_pred, squared=False))
                    logs['valid_metric'].append(mean_absolute_percentage_error(y_val, y_pred))

            sub_features_ind = sorted(
                np.random.choice(np.arange(m), int(self.feature_subsample_size * m), replace=False))
            subset_X = X[:, sub_features_ind]
            self.indeces_for_trees.append(sub_features_ind)

            bootstrap_indexes = np.random.randint(0, n, n)
            bootstrap_X = subset_X[bootstrap_indexes]
            bootstrap_y = y[bootstrap_indexes]

            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(bootstrap_X, bootstrap_y)
            self.trees.append(tree)

        if logging:
            logs['time'] = time.time() - logs['time']
            return logs

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        s = np.zeros(X.shape[0])
        for i, tree in enumerate(self.trees):
            s += tree.predict(X[:, self.indeces_for_trees[i]])
        return s / len(self.trees) if len(self.trees) > 0 else s


class GradientBoostingMSE:
    def __init__(
            self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
            **trees_parameters
    ):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use alpha * learning_rate instead of alpha

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use one-third of all features.
        """
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size \
            if feature_subsample_size is not None else 1. / 3
        self.params = trees_parameters

    @staticmethod
    def loss_gradients(targets, predictions):
        gradients = (predictions - targets)
        return gradients

    def fit(self, X, y, X_val=None, y_val=None, logging=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        n, m = X.shape
        self.trees = []
        self.indeces_for_trees = []
        self.alphas = []

        if logging:
            logs = {'train_loss': [], 'train_metric': [], 'valid_loss': [], 'valid_metric': [], 'time': time.time()}

        ensemble_preds = np.zeros(n)
        for _ in range(self.n_estimators):

            if logging:
                y_pred = self.predict(X)
                logs['train_loss'].append(mean_squared_error(y, y_pred, squared=False))
                logs['train_metric'].append(mean_absolute_percentage_error(y, y_pred))
                if X_val is not None and y_val is not None:
                    y_pred = self.predict(X_val)
                    logs['valid_loss'].append(mean_squared_error(y_val, y_pred, squared=False))
                    logs['valid_metric'].append(mean_absolute_percentage_error(y_val, y_pred))

            sub_features_ind = np.random.choice(np.arange(m), int(self.feature_subsample_size * m), replace=False)
            subset_X = X[:, sub_features_ind]
            self.indeces_for_trees.append(sub_features_ind)

            gradients = self.loss_gradients(y, ensemble_preds)
            tree = DecisionTreeRegressor(max_depth=self.max_depth).fit(subset_X, -gradients)

            self.trees.append(tree)
            cur_tree_preds = tree.predict(subset_X)
            alpha = minimize_scalar(lambda x: np.sum(ensemble_preds + x * cur_tree_preds - y) ** 2).x
            self.alphas.append(alpha)

            ensemble_preds = ensemble_preds + alpha * self.lr * cur_tree_preds

        if logging:
            logs['time'] = time.time() - logs['time']
            return logs

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros(X.shape[0])
        if len(self.trees) > 0:
            for i, tree in enumerate(self.trees):
                ans += self.alphas[i] * self.lr * tree.predict(X[:, self.indeces_for_trees[i]])

        return ans