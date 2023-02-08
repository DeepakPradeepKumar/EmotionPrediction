import numpy as np
import pandas as pd
#import os
#import sys
#import matplotlib.pyplot as plt
#import joblib


df = pd.read_csv('D:/Brainwave/TrainingDatafull.csv')

X = df.iloc[:,[1,2]].values
Y = df.iloc[:,3].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

#labelencoder = LabelEncoder()

#X[:,0] = labelencoder.fit_transform(X[:,0])
onehotencoder = ColumnTransformer([('Type', OneHotEncoder(),[0])] , remainder="passthrough")
X = onehotencoder.fit_transform(X)

#X[:,4] = (X[:,4] - X[:,4].mean() / X[:,4].std())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 101)	

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

######## RANDOM FOREST #############
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', max_depth=2, min_weight_fraction_leaf = 0.2, random_state = 0)
#classifier.fit(X_train, y_train)


###### KERNEL SVM ##############
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)



####### XGBOOST ################

from sklearn.grid_search import GridSearchCV
predictors = [x for x in Xtrain.columns if x not in [Ytrain]]
xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, Xtrain, predictors)




#joblib.dump(classifier, 'EmotionClassification.pkl')
# Predicting the Test set results
y_pred = classifier.predict(X_test)

def maxcount(y_pred):
	uniq, count = np.unique(y_pred, return_counts = True)
	maxcount = 0
	for i in range(len(uniq)):
		if count[i] > maxcount:
			maxcount = count[i]
			unique = uniq[i]

	return unique

print(maxcount(y_pred))

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

