import numpy as np
import statistics as st
import distance as my_dist
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, List, Dict
from extended_enum import __ExtendedEnum


class Method(__ExtendedEnum):
    own = 'my_own'
    kd_tree = 'kd_tree'
    ball_tree = 'ball_tree'
    brute = 'brute'


class Metric(__ExtendedEnum):
    euclidean = 'euclidean'
    cosine = 'cosine'


class KNNClassifier:

    def __init__(self, k: int, strategy: str, metric: str, weights: bool = False, test_block_size: int = None):
        self.__y_train = None
        self.__X_train = None
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.is_weighted = weights
        self.__epsilon = 10 ** (-5)
        self.test_block_size = test_block_size

        if strategy not in Method.list():
            raise TypeError

        if metric not in Metric.list():
            raise TypeError

        if test_block_size and test_block_size <= 0:
            raise TypeError

    def fit(self, X: np.array, y: np.array) -> None:
        if self.strategy == Method.own.value:
            self.__X_train = X
        elif self.strategy == Method.brute.value:
            self.__brute_fit(X)
        elif self.strategy == Method.kd_tree.value or self.strategy == Method.ball_tree.value:
            self.__create_tree(X)

        self.__y_train = y

    def find_kneighbors(self, X: np.array, return_distance: bool = False) -> Tuple[np.array, np.array]:
        if self.strategy == Method.kd_tree.value or self.strategy == Method.ball_tree.value:
            return self.__tree.kneighbors(X, return_distance=return_distance)
        elif self.strategy == Method.brute.value:
            return self.__brute_find_kneighbors(X, return_distance=return_distance)
        elif self.strategy == Method.own.value:
            return self.__own_find_kneighbors(X, return_distance=return_distance)

    def predict(self, X: np.array) -> np.array:
        y = np.tile(self.__y_train, (X.shape[0], 1))

        distances, indexes = self.find_kneighbors(X, True)
        y = y[np.arange(len(indexes)), indexes.T].T

        if not self.is_weighted:
            return np.apply_along_axis(func1d=st.mode, axis=1, arr=y)

        votes = self.__get_votes(dist=distances)

        uni_labels = np.unique(self.__y_train)

        ans = np.zeros((uni_labels.shape[0], votes.shape[0]))

        for i, label in enumerate(uni_labels):
            ans[i] = np.sum(votes * (y == label), axis=1)

        return np.argmax(ans.T, axis=1)

    def cv_predict(self, X, klist: List) -> Dict[int, np.array]:

        distances, neighbours = self.find_kneighbors(X, True)
        uni_labels = np.unique(self.__y_train)

        y = np.tile(self.__y_train, (neighbours.shape[0], 1))

        y = y[np.arange(len(neighbours)), neighbours.T].T

        cross_val_predict = {}
        ans = np.zeros((uni_labels.shape[0], neighbours.shape[0]))
        k_prev = 0

        if self.is_weighted:
            votes = self.__get_votes(dist=distances)
        else:
            votes = np.ones(neighbours.shape)

        for k in klist:
            for i, label in enumerate(uni_labels):
                ans[i] += np.sum(votes[:, k_prev:k] * (y[:, k_prev:k] == label), axis=1)

            cross_val_predict[k] = np.argmax(ans.T, axis=1)

            k_prev = k

        return cross_val_predict

    def __own_find_kneighbors(self, X: np.array, return_distance: bool = True) -> Tuple[np.array, np.array]:
        step = self.test_block_size if self.test_block_size else X.shape[0]

        neighbours_lst = []
        dist_lst = []

        for block in range(0, X.shape[0], step):
            X_block = X[block:block + step]
            if self.metric == Metric.euclidean.value:
                dist = my_dist.euclidean_distance(X_block, self.__X_train)
            elif self.metric == Metric.cosine.value:
                dist = my_dist.cosine_distance(X_block, self.__X_train)
            else:
                raise TypeError
            block_neighbours = np.argpartition(dist, kth=self.k, axis=1)[:, :self.k]
            block_dist = dist[np.arange(len(block_neighbours)), block_neighbours.T].T

            sorted_indexes = np.argsort(block_dist, axis=1)
            block_neighbours = block_neighbours[np.arange(len(sorted_indexes)), sorted_indexes.T].T
            block_dist = block_dist[np.arange(len(sorted_indexes)), sorted_indexes.T].T

            neighbours_lst.append(block_neighbours)
            dist_lst.append(block_dist)

        neighbours = np.concatenate(neighbours_lst)
        dist = np.concatenate(dist_lst)

        return (np.array(dist), np.array(neighbours)) if return_distance else np.array(neighbours)

    def __brute_find_kneighbors(self, X: np.array, return_distance: bool = True) -> Tuple[np.array, np.array]:
        step = self.test_block_size if self.test_block_size else X.shape[0]

        neighbours_lst = []
        dist_lst = []

        for block in range(0, X.shape[0], step):
            X_block = X[block:block + step]
            if return_distance:
                block_dist, block_neighbours = self.__brute.kneighbors(X_block, self.k, return_distance=return_distance)
            else:
                block_neighbours = self.__brute.kneighbors(X_block, self.k, return_distance=return_distance)
                block_dist = []
            
            neighbours_lst.append(block_neighbours)
            dist_lst.append(block_dist)

        neighbours = np.concatenate(neighbours_lst)
        if len(dist_lst)!=0:
            dist = np.concatenate(dist_lst)

        return (np.array(dist), np.array(neighbours)) if return_distance else np.array(neighbours)

    def __brute_fit(self, X: np.array) -> None:
        brute = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric,
                                 n_jobs=-1)
        self.__brute = brute.fit(X=X)

    def __create_tree(self, X: np.array) -> None:
        if self.metric != Metric.euclidean.value:
            raise TypeError

        self.__tree = NearestNeighbors(n_neighbors=self.k, algorithm=self.strategy, metric=self.metric,
                                       n_jobs=-1)

        self.__tree.fit(X=X)

    def __get_votes(self, dist: np.array) -> np.array:
        votes = 1 / (dist + self.__epsilon)
        return votes
