import numpy as np
import nearest_neighbors as nn
from extended_enum import __ExtendedEnum
from typing import Tuple, List, Dict


class Score(__ExtendedEnum):
    accuracy = 'accuracy'


def kfold(n: int, n_folds: int) -> np.array(Tuple[List, List]):
    indexes = np.arange(n)
    np.random.shuffle(indexes)
    indexes_set = set(indexes)
    folds = np.array_split(indexes, n_folds)
    for i, fold in enumerate(folds):
        folds[i] = np.array(list(indexes_set - set(fold))), fold
    return folds


def get_score(y1: np.array, y2: np.array, score_function: callable) -> float:
    return score_function(y1, y2)


def accuracy(y1: np.array, y2: np.array) -> float:
    if len(y1) != len(y2):
        raise TypeError
    return np.sum(y1 == y2) / len(y1)


def knn_cross_val_score(X: np.array, y: np.array, k_list: List[int], score: str,
                        cv: np.array(Tuple[List, List]) = None, **kwargs) -> Dict[int, List[float]]:
    if score not in Score.list():
        raise TypeError

    if not cv:
        print(cv)
        cv = kfold(len(y), 5)

    cross_val_score = {}
    for k in k_list:
        cross_val_score[k] = []

    for fold in cv:
        train, test = fold[0], fold[1]

        X_train, X_test = X[train], X[test]
        y_train, y_test = y[train], y[test]

        model = nn.KNNClassifier(k=k_list[-1], **kwargs)
        model.fit(X_train, y_train)
        predict_for_each_k = model.cv_predict(X_test, klist=k_list)
        for key in predict_for_each_k.keys():
            cross_val_score[key].append(get_score(y_test, predict_for_each_k[key], accuracy))

    return cross_val_score
