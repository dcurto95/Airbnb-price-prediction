from sklearn.neural_network import MLPRegressor
import time
import  numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score

def do_mlp_regressor(x_train,y_train,x_test,y_test):
    print("Training MLPRegressor...")
    since = time.time()
    mlp = MLPRegressor(activation='relu',
                       max_iter=1000, verbose=True)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    # MSE, R2
    mse = mean_squared_error(np.exp(y_test), np.exp(y_pred))
    r2 = r2_score(y_test, y_pred)
    print("done in {:.3f}s".format(time.time() - since))
    print("Test R2 score: {:.2f}".format(r2))
    print("Test MSE: {:.2f}".format(mse))
