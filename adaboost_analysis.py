import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.cross_validation import (ShuffleSplit, LeaveOneOut,
                                      cross_val_score)
from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import make_pipeline
import os

data_path = "/Users/au194693/projects/hyp_struct/data"
# data_path = "/projects/MINDLAB2011_33-MR-high-order-cogn/" +\
#             "scratch/Hypnoproj/MJ/data"
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

cv = ShuffleSplit(len(y), test_size=0.2)
loo = LeaveOneOut(len(y))

<<<<<<< HEAD
grid_estimators = []
scores_list = []

for train_cv, test_cv in cv:
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

    scores_list.append(
        cross_val_score(
            ada_cv, X[test_cv], y[test_cv], scoring="neg_mean_squared_error"))
=======
ada = AdaBoostRegressor()

adaboost_params = {
    "n_estimators": np.arange(2, 80, 1),
    "learning_rate": np.arange(0.1, 1.1, 0.1)
}

grid = GridSearchCV(
    ada, param_grid=adaboost_params, cv=loo, verbose=1, n_jobs=2,
    scoring="mean_squared_error")
grid.fit(X, y)

ada_cv = grid.best_estimator_

scores = cross_val_score(ada_cv, X, y, cv=loo, scoring="mean_squared_error")

# XGboost
cv_params = {
    "learning_rate": np.arange(0.1, 1.1, 0.1),
    "max_depth": [1, 2, 3, 4, 5, 6, 7],
    "n_estimators": np.arange(10, 86, 2)
}

grid = GridSearchCV(
    xgb.XGBClassifier(),
    cv_params,
    # scoring='accuracy',
    cv=cv,
    n_jobs=2,
    verbose=1)
grid.fit(X, y_group)
xgb_cv = grid.best_estimator_
>>>>>>> 2a2ea1b30914d0b9509e5b8da99042d22d9a7328
