import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor


def do_mlp_regressor(x_train, y_train, x_test, y_test):
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50),
                       learning_rate_init=0.01,
                       activation='tanh',
                       max_iter=1000, verbose=False)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)

    y_pred[(y_pred < 0)] = 0
    y_pred[(y_pred > 10)] = np.max(y_pred[(y_pred < 10)])

    # MSE, R2
    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)

    return mse, r2
