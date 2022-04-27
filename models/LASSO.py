from sklearn.linear_model import Lasso

class LASSO():
    def __init__(self) -> None:
        super(LASSO, self).__init__()
        self.lasso = Lasso(alpha= 1)


    def train(self, x_train, y_train):
        #history = self.lasso.fit(x_train, y_train, epoch=X, batch_size=Y)
        self.lasso.fit(x_train, y_train)

    def test(self, x_test):
        return self.lasso.predict(x_test)
