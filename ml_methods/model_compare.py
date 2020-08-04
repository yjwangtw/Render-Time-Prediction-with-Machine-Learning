#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:04:10 2017

@author: AnjieZheng
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.svm import SVC

from sklearn.metrics import r2_score

import numpy as np


# Read dataset from csv
df1 = pd.read_csv('features_1.csv')
df2 = pd.read_csv('features_2.csv')
df3 = pd.read_csv('features_3.csv')
frames = [df1, df2, df3]
df = pd.concat(frames)

# Specify Render engine
df = df[df['Render engine'] == 'BLENDER_RENDER']
df_tmp = df

# Specify testing scene
df = df_tmp[df_tmp['File'] != 'Calisma_1']
df_test = df_tmp[df_tmp['File'] == 'Calisma_1']

# Spliting x, y
feature_List = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30]
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
                     
#labelencoder_X_1 = LabelEncoder()
#X_test[:, 0] = labelencoder_X_1.fit_transform(X_test[:, 0])
#onehotencoder = OneHotEncoder(categorical_features = [0])
#X_test = onehotencoder.fit_transform(X_test).toarray()

#feature_List = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30]
#X = df.iloc[:, feature_List].values
#y = df.iloc[:, 29].values
#           
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# Standard Scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Construct Stacking Regressor
gdb = GradientBoostingRegressor(max_depth=1,n_estimators=15)
rf = RandomForestRegressor(n_estimators=25,max_features=15)
etc = ExtraTreesRegressor()
lr = LinearRegression()
knn = KNeighborsRegressor()
svr_rbf = SVR(kernel='rbf',C=1,epsilon=0.1)
svr_lin = SVR(kernel='linear')
stregr = StackingRegressor(regressors=[gdb, rf, svr_rbf], meta_regressor=lr)

# load dataset
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('KNN', KNeighborsRegressor()))
models.append(('SVR', SVR(kernel='rbf',C=1,epsilon=0.1)))
models.append(('GDB', GradientBoostingRegressor(max_depth=1,n_estimators=15)))
models.append(('RF', RandomForestRegressor(n_estimators=25,max_features=15)))
models.append(('ETC', ExtraTreesRegressor()))
models.append(('BC', BaggingRegressor()))
models.append(('STREGR', stregr))
# evaluate each model in turn
results = []
names = []
scoring = 'r2'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='r2')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    model.fit(X_train, y_train)
    t = model.predict(X_test)

    print(msg+" Test score: "+str(r2_score(y_test,t)))

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
ax = sns.boxplot(x=['KNN','SVR','GDB','RF','ETC', 'BC', 'STREGR'],y=results)
ax.set_xticklabels(names)
plt.show()
# fig.savefig('plot/algorithm_compare.png', format='png', dpi=1200)

# Draw learning curve of each models
import learningcurve
for name, model in models:
    title = "Learning Curves ("+name+")"
    estimator = model
    p1 = learningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.1, 1.01), cv=kfold, n_jobs=4)



