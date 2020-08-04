#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 16:29:47 2017

@author: AnjieZheng
"""

from mlxtend.regressor import StackingCVRegressor

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

RANDOM_SEED = 42

df1 = pd.read_csv('features_1.csv')
df2 = pd.read_csv('features_2.csv')
df3 = pd.read_csv('features_3.csv')
frames = [df1, df2, df3]
df = pd.concat(frames)
df = df[df['Render engine'] == 'BLENDER_RENDER']

df_tmp = df

df = df_tmp[~df_tmp['File'].isin(['Calisma_2','Ana_2','Bambo_House_2'])]
df_test = df_tmp[df_tmp['File'].isin(['Calisma_2','Ana_2','Bambo_House_2'])]

feature_List = [3,4,5,6,7,8,9,10,11,12,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30]
X_train = df.iloc[:, feature_List].values
y_train = df.iloc[:, 29].values
                 
# Encoding categorical data    
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_0 = LabelEncoder()
#X_train[:, 0] = labelencoder_X_0.fit_transform(X_train[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X_train = onehotencoder.fit_transform(X_train).toarray()
                 
X_test = df_test.iloc[:, feature_List].values
y_test = df_test.iloc[:, 29].values


rf = RandomForestRegressor(random_state=RANDOM_SEED)
svr = SVR(kernel='rbf')
svr_lin = SVR(kernel='linear')
gbr = GradientBoostingRegressor()
lr = LinearRegression()

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors=(rf, svr, gbr),
                            meta_regressor=rf,use_features_in_secondary=True)


grid = GridSearchCV(
    estimator=stack, 
    param_grid={
        'randomforestregressor__n_estimators' : [25],
        'randomforestregressor__max_features' : [15],
        'svr__C' : [1],
        'svr__epsilon' : [0.1],
        'gradientboostingregressor__max_depth': [1],
        'gradientboostingregressor__n_estimators': [5,10,15,20,25,30,35,40,45,50], #15
        'meta-randomforestregressor__n_estimators': [200],
        'meta-randomforestregressor__max_features': [10],
    }, 
    cv=10,
    refit=True
)

grid.fit(X_train, y_train)  

# Print
cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))
    if r > 10:
        break
print('...')

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)


#rf1 = RandomForestRegressor(random_state=RANDOM_SEED,n_estimators=25,max_features=15)
#rf2 = RandomForestRegressor(random_state=RANDOM_SEED,n_estimators=200,max_features=10)
#svr = SVR(kernel='rbf',C=1,epsilon=0.1)
#gbr = GradientBoostingRegressor(n_estimators=15,max_depth=1)
#stack_best = StackingCVRegressor(regressors=(rf1, svr, gbr),
#                            meta_regressor=rf2,use_features_in_secondary=True)
#
#stack_best.fit(X_train, y_train)
#t = stack_best.predict(X_test)
#print("Stack Test score: "+str(r2_score(y_test,t)))










