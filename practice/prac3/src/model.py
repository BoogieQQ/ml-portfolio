from ensembles import RandomForestMSE, GradientBoostingMSE


class Model:
    def __init__(self, params):
        self.model_id = int(params[0])
        self.n_estimators = int(params[1])
        self.max_depth = int(params[2])
        self.feature_subsample_size = float(params[3])
        if self.model_id == 0:
            self.learning_rate = None
            self.ensemble = RandomForestMSE(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            feature_subsample_size=self.feature_subsample_size)
        if self.model_id == 1:
            self.learning_rate = float(params[4])
            self.ensemble = GradientBoostingMSE(n_estimators=self.n_estimators, max_depth=self.max_depth,
                                                feature_subsample_size=self.feature_subsample_size,
                                                learning_rate=self.learning_rate)

    def fit(self, X, y, X_val=None, y_val=None):
        return self.ensemble.fit(X, y, X_val, y_val, logging=True)

    def predict(self, X):
        return self.ensemble.predict(X)

    def get_info(self):
        model_type = 'Градиентный бустинг' if self.model_id else 'Случайный лес'
        lr = self.learning_rate if self.learning_rate is not None else '-'
        return [model_type, self.n_estimators, self.max_depth, self.feature_subsample_size, lr]
