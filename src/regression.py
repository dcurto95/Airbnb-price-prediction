from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)
    error = mean_squared_error(y_test, y_pred)
    print('Mean squared error: %.2f' % error)
    print('Coefficient of determination: %.2f'
          % r2_score(y_test, y_pred))
