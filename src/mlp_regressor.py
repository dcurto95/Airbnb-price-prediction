from sklearn.neural_network import MLPRegressor
import time
import  numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score

def do_mlp_regressor(x_train,y_train,x_test,y_test):
    print("Training MLPRegressor...")
    since = time.time()
    mlp = MLPRegressor(hidden_layer_sizes=(50, 50, 50, 50),
                        learning_rate_init=0.01,
                        activation='tanh',
                       max_iter=1000, verbose=True)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    # MSE, R2
    mse = mean_squared_error(y_test, np.exp(y_pred))
    r2 = r2_score(np.log(y_test), y_pred)
    print("done in {:.3f}s".format(time.time() - since))
    print("Test R2 score: {:.2f}".format(r2))
    print("Test MSE: {:.2f}".format(mse))
