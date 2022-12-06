import sklearn
import pandas as pd
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as rf
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier as xgb
import sys
from model import MLP
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda



device = 'cpu'

# Load dataset
data = './data/test crank-ms.csv'
df = pd.read_csv(data, encoding='latin1')
df = df.sample(frac=1).reset_index(drop=True)  # shuffle rows
print('length of columns = ', len(df.columns))

X = df.iloc[:,1:]
y = df.iloc[:,0]

# data split iter
sss = ShuffleSplit(n_splits=100, test_size=0.4)
sss.get_n_splits(X, y)


# MLP
# model = MLP((32,))

# lambda_l1 = 0.001291549665014884
# lambda_l2 = 0.000774263682681127

# parameters = {
#     'num_epochs': [32,64],
#     'batch_size': [16],
#     'lambda_l1': np.geomspace(lambda_l1 / 2, lambda_l1 * 2, 8),
#     'lambda_l2': np.geomspace(lambda_l2 / 2, lambda_l2 * 2, 8),
# }

# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     scoring = make_scorer(
#         matthews_corrcoef,
#         greater_is_better = True,
#     ),
#     n_jobs = 12,
#     cv = sss.split(X, y),
#     verbose = 4
# )

# grid_search.fit(X, y)
# print('Best score (MCC):\t', grid_search.best_score_)
# print('Best parameters:\t', grid_search.best_params_)


#XGB
# model = xgb(
#     objective= 'binary:logistic',
#     nthread=1,
#     seed=100,
#     eval_metric='logloss',
# )

# parameters = {
# 'max_depth': [1],
# 'n_estimators': [50], 
# 'learning_rate': np.linspace(0.6, 0.1, 6),
# 'reg_alpha': [0.5, 0.05],
# 'reg_lambda': [0.5, 0.05],
# 'colsample_bytree': np.linspace(1, 0.2, 5),
# # 'min_child_weight': [10,0, 1],
# # 'min_split_loss': [10,0, 1],
# # 'subsample': np.linspace(1, 0.2, 5),
# }

# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     scoring = make_scorer(
#         matthews_corrcoef,
#         greater_is_better = True,
#     ),
#     n_jobs = 12,
#     cv = sss.split(X, y),
#     verbose = 4
# )

# grid_search.fit(X, y)
# print('Best score (MCC):\t', grid_search.best_score_)
# print('Best parameters:\t', grid_search.best_params_)


#RF
# model = rf(
# )

# parameters = {
# 'n_estimators': [25, 50, 100, 250, 500],
# 'min_samples_split': [2, 3, 4, 5],
# 'max_features': ['auto', 'sqrt', 'log2'],
# 'criterion': ['gini', 'entropy'],
# }

# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     scoring = make_scorer(
#         matthews_corrcoef,
#         greater_is_better = True,
#     ),
#     n_jobs = 12,
#     cv = sss.split(X, y),
#     verbose = 4
# )

# grid_search.fit(X, y)
# print('Best score (MCC):\t', grid_search.best_score_)
# print('Best parameters:\t', grid_search.best_params_)


#Lr
# model = lr(
#         )

# parameters = {
# 'solver': ['saga', 'liblinear'],
# 'C': range(1, 1000, 60),
# 'penalty': ['l1'],
# }

# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     scoring = make_scorer(
#         matthews_corrcoef,
#         greater_is_better = True,
#     ),
#     n_jobs = 12,
#     cv = sss.split(X, y),
#     verbose = 4
# )

# grid_search.fit(X, y)
# print('Best score (MCC):\t', grid_search.best_score_)
# print('Best parameters:\t', grid_search.best_params_)


# #lda

# model = lda(
#         )

# parameters = {
# 'solver': ['lsqr'],
# 'shrinkage': ['auto', 'none', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# }

# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=parameters,
#     scoring = make_scorer(
#         matthews_corrcoef,
#         greater_is_better = True,
#     ),
#     n_jobs = 12,
#     cv = sss.split(X, y),
#     verbose = 4
# )

# grid_search.fit(X, y)
# print('Best score (MCC):\t', grid_search.best_score_)
# print('Best parameters:\t', grid_search.best_params_)


# #SVM

model = SVC(
    probability = True)

parameters = {
    'kernel': ['linear'],
    'C': [0.1, 0.25, 0.5, 0.75, 1, 5, 10, 25, 50, 75, 100],
    'gamma': [0.0001, 0.0005, 0.001, 
              0.005,
              0.01, 0.05,
              0.1, 0.25, 0.5, 0.75, 1]
    }

grid_search = GridSearchCV(
    estimator=model,
    param_grid=parameters,
    scoring = make_scorer(
        matthews_corrcoef,
        greater_is_better = True,
    ),
    n_jobs = 4,
    cv = sss.split(X, y),
    verbose = 4
)

grid_search.fit(X, y)
print('Best score (MCC):\t', grid_search.best_score_)
print('Best parameters:\t', grid_search.best_params_)
