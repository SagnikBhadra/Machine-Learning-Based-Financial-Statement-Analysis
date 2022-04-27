from sklearn.ensemble import RandomForestRegressor

class RandomForest:
    def __init__(self):
        #CART tree
        #num_tress_in_forest = 200
        #Consider depth of tree hyperparameter

        self.n_estimators = 200
        self.random_forest = RandomForestRegressor(n_estimators = self.n_estimators)

    def train(self, x_train, y_train):
        self.random_forest_fitted = self.random_forest.fit(x_train, y_train)

    def test(self, x_test):
        y_predictions = self.random_forest_fitted.predict(x_test)
        return y_predictions
        