from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR


def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)
    error = mean_squared_error(y_test, y_pred)
    print('Mean squared error: %.2f' % error)
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))

def svr(x_train, y_train, x_test, y_test):
    g = 0.06
    c = 100
    # Train the model
    reg_svr = SVR(kernel='rbf', gamma=g, C=c).fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg_svr.predict(x_test)
    # MSE, R2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
