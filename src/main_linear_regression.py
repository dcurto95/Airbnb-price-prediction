import time

import pandas as pd

import mlp_regressor
import regression


def do_regressions(x_train, y_train, x_test, y_test):
    print("\nMLP regressor:")
    since = time.time()
    mse_lr, r2_lr = mlp_regressor.do_mlp_regressor(x_train, y_train, x_test, y_test)
    print('\tMean squared error linear regression: %.2f' % mse_lr)
    print('\tCoefficient of determination linear regression: %.2f' % r2_lr)
    print("\tExecution time:", time.time() - since, "s")

    print("\nLinear regression:")
    since = time.time()
    mse_lr, r2_lr = regression.linear_regression(x_train, y_train, x_test, y_test)
    print('\tMean squared error linear regression: %.2f' % mse_lr)
    print('\tCoefficient of determination linear regression: %.2f' % r2_lr)
    print("\tExecution time:", time.time() - since, "s")

    print("\nSVR:")
    since = time.time()
    mse_svr, r2_svr = regression.svr(x_train, y_train, x_test, y_test)
    print('\tMean squared error SVR: %.2f' % mse_svr)
    print('\tCoefficient of determination SVR: %.2f' % r2_svr)
    print("\tExecution time:", time.time() - since, "s")


if __name__ == '__main__':
    # ---------- WITHOUT FUZZY & WITH NEIGHBOURHOOD ----------
    print("\nResults without fuzzy feature and with neighbourhood:")
    train_cleaned_df = pd.read_csv("../data/train_cleaned.csv")
    validation_cleaned_df = pd.read_csv("../data/validation_cleaned.csv")
    test_cleaned_df = pd.read_csv("../data/test_cleaned.csv")

    x_train = train_cleaned_df.drop(columns=['price']).to_numpy()
    x_validation = validation_cleaned_df.drop(columns=['price']).to_numpy()
    x_test = test_cleaned_df.drop(columns=['price']).to_numpy()

    y_train = train_cleaned_df['price'].to_numpy()
    y_validation = validation_cleaned_df['price'].to_numpy()
    y_test = test_cleaned_df['price'].to_numpy()

    do_regressions(x_train, y_train, x_test, y_test)

    # ---------- FUZZY & WITH NEIGHBOURHOOD  ----------
    print("\nResults with fuzzy feature and with neighbourhood:")
    train_fuzzy_df = pd.read_csv("../data/train_fuzzy.csv")
    validation_fuzzy_df = pd.read_csv("../data/validation_fuzzy.csv")
    test_fuzzy_df = pd.read_csv("../data/test_fuzzy.csv")

    x_train = train_fuzzy_df.drop(columns=['price']).to_numpy()
    x_validation = validation_fuzzy_df.drop(columns=['price']).to_numpy()
    x_test = test_fuzzy_df.drop(columns=['price']).to_numpy()

    y_train = train_fuzzy_df['price'].to_numpy()
    y_validation = validation_fuzzy_df['price'].to_numpy()
    y_test = test_fuzzy_df['price'].to_numpy()

    do_regressions(x_train, y_train, x_test, y_test)

    ################# WITHOUT NEIGHBOURHOOD ###################

    # ---------- WITHOUT FUZZY & WITHOUT NEIGHBOURHOOD ----------
    print("\nResults without fuzzy feature and without neighbourhood:")
    train_cleaned_df = pd.read_csv("../data/train_cleaned_wo_Neigh.csv")
    validation_cleaned_df = pd.read_csv("../data/validation_cleaned_wo_Neigh.csv")
    test_cleaned_df = pd.read_csv("../data/test_cleaned_wo_Neigh.csv")

    x_train = train_cleaned_df.drop(columns=['price']).to_numpy()
    x_validation = validation_cleaned_df.drop(columns=['price']).to_numpy()
    x_test = test_cleaned_df.drop(columns=['price']).to_numpy()

    y_train = train_cleaned_df['price'].to_numpy()
    y_validation = validation_cleaned_df['price'].to_numpy()
    y_test = test_cleaned_df['price'].to_numpy()

    do_regressions(x_train, y_train, x_test, y_test)

    # ---------- FUZZY & WITHOUT NEIGHBOURHOOD  ----------
    print("\nResults with fuzzy feature and without neighbourhood:")
    train_fuzzy_df = pd.read_csv("../data/train_fuzzy_wo_Neigh.csv")
    validation_fuzzy_df = pd.read_csv("../data/validation_fuzzy_wo_Neigh.csv")
    test_fuzzy_df = pd.read_csv("../data/test_fuzzy_wo_Neigh.csv")

    x_train = train_fuzzy_df.drop(columns=['price']).to_numpy()
    x_validation = validation_fuzzy_df.drop(columns=['price']).to_numpy()
    x_test = test_fuzzy_df.drop(columns=['price']).to_numpy()

    y_train = train_fuzzy_df['price'].to_numpy()
    y_validation = validation_fuzzy_df['price'].to_numpy()
    y_test = test_fuzzy_df['price'].to_numpy()

    do_regressions(x_train, y_train, x_test, y_test)
