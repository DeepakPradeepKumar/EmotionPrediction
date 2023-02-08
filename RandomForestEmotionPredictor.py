import os
import sys
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('D:\\Brainwave\\demo_data_happy2.csv')

X = df.iloc[:,[1,2]].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
onehotencoder = ColumnTransformer([('type', OneHotEncoder(),[0])] , remainder="passthrough")
X = onehotencoder.fit_transform(X)

#X[:,2] = (X[:,2] - X[:,2].mean() / X[:,2].std())

model1 = joblib.load('D:\\Brainwave\\EmotionClassificationmodel1.pkl')
model2 = joblib.load('D:\\Brainwave\\EmotionClassificationmodel2.pkl')
model3 = joblib.load('D:\\Brainwave\\EmotionClassificationmodel3.pkl')


pred1 = model1.predict(X)
pred2 = model2.predict(X)
pred3 = model3.predict(X)


#final_pred = (pred1+pred2+pred3)/3

# print("pred1 = ", pred1)
# print("pred2 = ", pred2)
# print("pred3 = ", pred3)

# print(pred1.shape)
# print(pred2.shape)
# print(pred3.shape)

finalpred = np.vstack([pred1, pred2 , pred3])
print(" final pred ", finalpred.shape)


uniq, count = np.unique(finalpred, return_counts = True)
# print(uniq, count)

# uniq, count = np.unique(pred1, return_counts = True)
# uniq, count = np.unique(pred2, return_counts = True)
# uniq, count = np.unique(pred3, return_counts = True)

# predcount = [pred1, pred2, pred3]

# for pred in range(len(predcount)):
# 	uniq, count = np.unique(pred, return_counts = True)
# 	count[i] += count[i]
# 	uniq[i] += uniq[i]


for i in range(len(uniq)):
	if uniq[i] == 0:
		count[i] = count[i]/5.5
	# elif uniq[i] == 1:
	# 	count[i] = count[i] + (count[i] / 3)
	elif uniq[i] == 2:
		count[i] = count[i]
	else:
		count[i] = count[i]

print(count)


# uniq, count = np.unique(final_pred, return_counts = True)
# count = [np.int(count[i]/5.5) if uniq[i] == 0 else count[i] for i in range(len(uniq))]
# print(count)
# print("final Pred ", final_pred)
print(count)

def maxcount(Y):
	maxcount = 0
	for i in range(len(uniq)):
		if count[i] > maxcount:
			maxcount = count[i]
			unique = uniq[i]

	return unique,count, maxcount

uniq, count = np.unique(np.round(finalpred), return_counts = True)
print(((uniq, count)))

print(maxcount(finalpred))



