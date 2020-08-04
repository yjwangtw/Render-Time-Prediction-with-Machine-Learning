#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 01:28:36 2017

@author: AnjieZheng
"""
import numpy as np
import pandas as pd

df1 = pd.read_csv('features_1.csv')
df2 = pd.read_csv('features_2.csv')
df3 = pd.read_csv('features_3.csv')
frames = [df1, df2, df3]
df = pd.concat(frames)
df_tmp = df

df = df_tmp[df_tmp['File'] != 'BlenderBattery']
df_test = df_tmp[df_tmp['File'] == 'BlenderBattery']

feature_List = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30]
X_train = df.iloc[:, feature_List].values
y_train = df.iloc[:, 29].values
                 
X_test = df_test.iloc[:, feature_List].values
y_test = df_test.iloc[:, 29].values

#feature_List = [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30]
#X = df.iloc[:, feature_List].values
#y = df.iloc[:, 29].values
#           
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# SVM
from sklearn import svm
from sklearn.metrics import r2_score
model_svm = svm.SVR(C=100,epsilon=0.1,kernel='rbf')

model_svm.fit(X_train, y_train)
y_pred = model_svm.predict(X_train)
print ("SVM Train: "+str(r2_score(y_train,y_pred)))

t = model_svm .predict(X_test)
print ("SVM Test: "+str(r2_score(y_test,t)))

# SVM Gridsearch
#from sklearn.grid_search import GridSearchCV
#Cs = [0.1, 1, 10,100, 1000]
#epsilons = [0.001, 0.01, 0.1, 1]
#param_grid = {'C': Cs, 'epsilon' : epsilons}
#grid_search = GridSearchCV(svm.SVR(kernel='rbf'), param_grid, n_jobs=-1)
#grid_search.fit(X_train, y_train)
#print(grid_search.best_score_)
#print(grid_search.best_params_)

# Random Forest
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor

num_dataset=X_train.shape[0] #HJ: I hard code the number of lines of the dataset
rng = np.random.RandomState(num_dataset)#32704 is the size of dataset

Ns = [100,120]
Max_Features = [27,20,15,10,5]
avg_scores=[]
tmp_n=[]
tmp_f=[]

for max_f in Max_Features:
    for N in Ns:
        kf = KFold(y_train.shape[0], n_folds=10, shuffle=True, random_state=rng)
        scores = []
        for train_index, test_index in kf:
            rf = RandomForestRegressor(n_estimators=N, max_features=max_f)
            rf.fit(X_train[train_index], y_train[train_index])
            predictions = rf.predict(X_train[test_index])
            actuals = y_train[test_index]
            scores.append(r2_score(actuals, predictions))
            # print scores
        print ("Ns: "+str(N)+", Features: "+str(max_f)+", Average score: " + str(np.average(scores)))
        avg_scores.append(np.average(scores));
        tmp_n.append(N);
        tmp_f.append(max_f);

print ("Best:")
best_index=avg_scores.index(max(avg_scores))
best_f=tmp_f[best_index]
best_n=tmp_n[best_index]
print ("Score: "+str(avg_scores[best_index])+", Ns: "+str(best_n)+", F: "+str(best_f))

rf = RandomForestRegressor(n_estimators=best_n, max_features=best_f)
rf.fit(X_train, y_train)

t = rf.predict(X_test)
print ("RF Test Test: "+str(r2_score(y_test,t)))

#######
from sklearn.cross_validation import KFold
from sklearn import svm

num_dataset=X_train.shape[0] #HJ: I hard code the number of lines of the dataset
rng = np.random.RandomState(num_dataset)#32704 is the size of dataset

coe_C = [0.1, 1, 10, 100, 1000]
coe_E = [0.1, 0.001]
avg_scores=[]
tmp_c=[]
tmp_e=[]

for coe_e in coe_E:  
    for coe_c in coe_C:
        kf = KFold(y_train.shape[0], n_folds=10, shuffle=True, random_state=rng)
        scores = []
        for train_index, test_index in kf:
            model_svm = svm.SVR(C=coe_c,epsilon=coe_e,kernel='rbf')
            model_svm.fit(X_train[train_index], y_train[train_index])
            predictions = model_svm.predict(X_train[test_index])
            actuals = y_train[test_index]
            scores.append(r2_score(actuals, predictions))
            # print scores
        print ("C: "+str(coe_c)+",E: "+str(coe_e)+", Average score: " + str(np.average(scores)))
        avg_scores.append(np.average(scores));
        tmp_c.append(coe_c);
        tmp_e.append(coe_e);
    
print ("Best:")
best_index=avg_scores.index(max(avg_scores))
best_c=tmp_c[best_index]
best_e=tmp_e[best_index]
print ("C:"+str(best_c))
print ("E:"+str(best_e))
print ("SVM Train Score: "+str(avg_scores[best_index]))

model_svm = svm.SVR(C=best_c,epsilon=best_e,kernel='rbf')
model_svm.fit(X_train, y_train)

t = model_svm .predict(X_test)
print ("SVM Test Test: "+str(r2_score(y_test,t)))
#######

y = (y // 10)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# SVM (Classification)
from sklearn import svm
from sklearn.metrics import accuracy_score
model_svm = svm.SVC(C=100,kernel='rbf',probability=True)
svm.verbose = True
model_svm .fit(X_train, y_train)
y_pred = model_svm .predict(X_train)
print ("SVM Train: "+str(accuracy_score(y_train,y_pred)))

t = model_svm .predict(X_test)
print ("SVM Test: "+str(accuracy_score(y_test,t)))

# RF (RandomForest)
from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(n_estimators=120, max_features=5)
model_rf.fit(X_train, y_train)
predictions = model_rf.predict(X_train)
print ("RF Train: "+str(accuracy_score(y_train,predictions)))

predictions = model_rf.predict(X_test)
print ("RF Test: "+str(accuracy_score(y_test,predictions)))

output_svm_train = model_svm.predict_proba(X_train)
output_rf_train = model_rf.predict_proba(X_train)
output_train = np.concatenate((output_svm_train,output_rf_train),axis=1)

output_svm_test = model_svm.predict_proba(X_test)
output_rf_test = model_rf.predict_proba(X_test)
output_test = np.concatenate((output_svm_test,output_rf_test),axis=1)

from sklearn.linear_model import LogisticRegression   
classifier = LogisticRegression()
classifier.fit(output_train, y_train)
predictions = classifier.predict(output_test)
print ("Final Test: "+str(accuracy_score(y_test,predictions)))

##########
##########
######













import learningcurve
from sklearn.model_selection import ShuffleSplit


title = "Learning Curves (RandomForestRegressor)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)

estimator = rf = RandomForestRegressor(n_estimators=best_n, max_features=best_f)
p1 = learningcurve.plot_learning_curve(estimator, title, X_train, y_train, ylim=(0.1, 1.01), cv=10, n_jobs=4)
p1.grid()
p1.savefig('LearningCurve_rf.png', format='png', dpi=1200)


title = "Learning Curves (SVM, RBF, C = 100)"
# SVC is more expensive so we do a lower number of CV iterations:

estimator = svm.SVR(C=best_c,epsilon=best_e,kernel='rbf')
p2 = learningcurve.plot_learning_curve(estimator, title, X_train, y_train, (0.5, 1.01), cv=10, n_jobs=4)


from sklearn.learning_curve import validation_curve
train_scores, test_scores = validation_curve(estimator=svm.SVR(), X=X_train, y=y_train, param_name='C',param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

title = "Learning Curves (SVM, k=rbf)"
plt.grid()
plt.title(title)
plt.xscale('log')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.1, 1.0])
plt.tight_layout()
plt.savefig('coe_curve_svm.png', format='png', dpi=1200)
plt.show()


######係數學習曲線RF

from sklearn.learning_curve import validation_curve
train_scores, test_scores = validation_curve(estimator=RandomForestRegressor(n_estimators=best_n), X=X_train, y=y_train, param_name='max_features',param_range=Max_Features, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(Max_Features, train_mean,
         color='blue', marker='o',
         markersize=5, label='training accuracy')

plt.fill_between(Max_Features,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(Max_Features, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='validation accuracy')

plt.fill_between(Max_Features,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

title = "Learning Curves (RandomForestRegressor)"
plt.grid()
plt.title(title)
plt.xlabel('Max Features')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.tight_layout()
plt.savefig('coe_curve_rf.png', format='png', dpi=1200)
plt.show()


