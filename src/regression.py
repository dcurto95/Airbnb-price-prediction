from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)
    pc = pearsonr(y_test, y_pred)
    return pc
