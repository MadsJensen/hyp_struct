import os
import pandas as pd
import numpy as np

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import (ShuffleSplit, cross_val_score)
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error

# data_path = "/Users/au194693/projects/hyp_struct/data"
data_path = "/projects/MINDLAB2011_33-MR-high-order-cogn/" +\
            "scratch/Hypnoproj/MJ/data"
os.chdir(data_path)

all_data = pd.read_csv("all_data.csv")
data_selected = all_data[["id", "ThickAvg_lh", "ThickAvg_rh", "shss"]]

X = []
y = []
subjects = data_selected.id.unique()

for sub in subjects:
    tmp = data_selected[data_selected.id == sub]
    X.append(tmp[["ThickAvg_lh", "ThickAvg_rh"]].get_values().reshape(-1))
    y.append(tmp.shss.get_values()[0])

X = np.asarray(X)
y = np.asarray(y)

X = X[2:, :-1]
y = y[2:]

cv = ShuffleSplit(len(y), test_size=0.15)

grid_estimators = []
scores_list = []

for train_cv, test_cv in cv.spilt(X, y):
    # Setup grid parameters
    ada = AdaBoostRegressor()
    adaboost_params = {
        "n_estimators": np.arange(2, 80, 1),
        "learning_rate": np.arange(0.1, 1.1, 0.1)
    }

    # Make grid search
    grid = GridSearchCV(
        ada,
        param_grid=adaboost_params,
        scoring="mean_squared_error",
        verbose=1,
        n_jobs=1)
    grid.fit(X[train_cv], y[train_cv])

    # Select winning params
    ada_cv = grid.best_estimator_
    grid_estimators.append(ada_cv)

    y_pred = ada_cv.predict(X[test_cv])

    scores_list.append(mean_squared_error(y[test_cv], y_pred))

# make model based on the mean
ada_learning_rate = np.median(
    np.asarray([est.learning_rate for est in grid_estimators]))

ada_n_estimators = int(
    np.median(np.asarray([est.n_estimators for est in grid_estimators])))

ada_mean = AdaBoostRegressor(
    n_estimators=ada_n_estimators, learning_rate=ada_learning_rate)

scores = cross_val_score(ada_mean, X, y, cv=cv, scoring="mean_squared_error")
