import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix



def linear_regression(x_train, y_train, x_test, y_test):
    # Train the model
    reg = LinearRegression().fit(x_train, y_train)
    # Predict using x_test
    y_pred = reg.predict(x_test)
    # Accuracy
    a = accuracy_score(y_test, y_pred)
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    return a, cm

