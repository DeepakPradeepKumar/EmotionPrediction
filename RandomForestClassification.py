import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.externals import joblib


df = pd.read_csv('D:\\Brainwave\\TrainingDatafull.csv')

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

######### RANDOM FOREST ##############
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', max_depth=2,  random_state = 1)
#classifier.fit(X_train, y_train)
#cl = classifier.score(X_test, y_test)
#print(cl)


## ENSEMBLING TECHNIQUES
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

DTModel = tree.DecisionTreeClassifier()
KNNModel = KNeighborsClassifier()
LogRegModel = LogisticRegression()

model1 = DTModel.fit(X_train, y_train)
model2 = KNNModel.fit(X_train, y_train)
model3 = LogRegModel.fit(X_train, y_train)

pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
pred3 = model3.predict(X_test)

final_pred = (pred1+pred2+pred3)/3


joblib.dump(model1,  'EmotionClassificationmodel1.pkl')
joblib.dump(model2,  'EmotionClassificationmodel2.pkl')
joblib.dump(model3,  'EmotionClassificationmodel3.pkl')

# Predicting the Test set results
#y_pred = classifier.predict(X_test)

def maxcount(y_pred):
	uniq, count = np.unique(y_pred, return_counts = True)
	maxcount = 0
	for i in range(len(uniq)):
		if count[i] > maxcount:
			maxcount = count[i]
			unique = uniq[i]

	return unique

#print(maxcount(final_pred))
#print(maxcount(y_test))

#print(maxcount(y_pred))

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

