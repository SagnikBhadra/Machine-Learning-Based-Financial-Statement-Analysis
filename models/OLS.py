from sklearn.linear_model import LinearRegression

class LASSO(nn.Module):
    def __init__(self) -> None:
        super(LASSO, self).__init__()
        self.ols = LinearRegression()


    def train(self, x_train, y_train):
        self.ols.fit(x_train, y_train)

    def test(self, x_test):
        return self.ols.predict(x_test)
