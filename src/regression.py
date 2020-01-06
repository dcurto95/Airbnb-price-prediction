import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)

    y_pred[(y_pred < 0)] = 0
    y_pred[(y_pred > 10)] = np.max(y_pred[(y_pred < 10)])

    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def svr(x_train, y_train, x_test, y_test):
    # Train the model
    reg_svr = SVR(kernel='rbf', gamma='scale').fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg_svr.predict(x_test)

    y_pred[(y_pred < 0)] = 0
    y_pred[(y_pred > 10)] = np.max(y_pred[(y_pred < 10)])
    
    # MSE, R2
    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)
    return mse, r2
