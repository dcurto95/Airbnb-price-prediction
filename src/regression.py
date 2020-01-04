import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)
    mse = mean_squared_error(y_test, np.exp(y_pred))
    r2 = r2_score(np.log(y_test), y_pred)
    return mse, r2


def svr(x_train, y_train, x_test, y_test):
    # Train the model
    reg_svr = SVR(kernel='rbf', gamma='scale').fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg_svr.predict(x_test)
    # MSE, R2
    mse = mean_squared_error(y_test, np.exp(y_pred))
    r2 = r2_score(np.log(y_test), y_pred)
    return mse, r2
