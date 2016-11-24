import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import (ShuffleSplit, cross_val_score)
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error

import xgboost as xgb

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

for train_cv, test_cv in cv:
    # Setup grid parameters
    xgb_reg = xgb.XGBRegressor()
    adaboost_params = {
        "n_estimators": np.arange(2, 80, 1),
        "learning_rate": np.arange(0.1, 1.1, 0.1),
        "max_depth": np.arange(1, 8, 1)
    }

    # Make grid search
    grid = GridSearchCV(
        xgb_reg,
        param_grid=adaboost_params,
        scoring="mean_squared_error",
        verbose=1,
        n_jobs=1)
    grid.fit(X[train_cv], y[train_cv])

    # Select winning params
    xgb_cv = grid.best_estimator_
    grid_estimators.append(xgb_cv)

    y_pred = xgb_cv.predict(X[test_cv])

    scores_list.append(mean_squared_error(y[test_cv], y_pred))
