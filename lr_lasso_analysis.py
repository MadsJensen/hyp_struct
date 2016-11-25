import os
import pandas as pd
import numpy as np

from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV

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
    lasso_cv = LassoCV()
    lasso_cv.fit(X[train_cv], y[train_cv])
    y_pred = lasso_cv.predict(X[test_cv])

    grid_estimators.append(lasso_cv)
    scores_list.append(mean_squared_error(y[test_cv], y_pred))


lr = np.asarray([foo.alphas_ for foo in grid_estimators])