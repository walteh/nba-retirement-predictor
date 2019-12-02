import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# y = pd.read_csv('./created/retire.csv')
df = pd.read_csv('./created/stats_no2020_noNull.csv')

info = df.iloc[:, :5]

# X = feature values, all the columns except the last column
X = df.iloc[:, 5:-1]

# y = target values, last column of the data frame
y = df.iloc[:, -1]
vinceIndex = df.loc[df['id']=='cartevi01-2010'].index[0]

vince = df.iloc[vinceIndex:vinceIndex+1, 5:-1]
# print(vince)

# filter out the applicants that got admitted
retired = df.loc[y == 1]

# filter out the applicants that din't get admission
not_retired = df.loc[y == 0]

# print(X)


from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = 0.25) 

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression(fit_intercept=False, class_weight=({0: 2, 1: 2})) 

# print(xtrain)

model.fit(xtrain, ytrain) 

r_sq = model.score(xtest, ytest)

y_pred = model.predict(xtest) 
vince_pred = model.predict_proba(vince) 

print('Vince: ', vince_pred)


# print(xtest)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 
