import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix
# y = pd.read_csv('./created/retire.csv')
df = pd.read_csv('./created/stats_no2020_noPercents.csv')

info = df.iloc[:, :5]

# X = feature values, all the columns except the last column
X = df.iloc[:, 5:-1]

# y = target values, last column of the data frame
y = df.iloc[:, -1]
vinceIndex = df.loc[df['id']=='anthoca01-2019'].index[0]
lebronIndex = df.loc[df['id']=='anthoca01-2019'].index[0]

# vince cartevi01
# melo anthoca01-2010


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

confusion = confusion_matrix(ytest, y_pred)

print('confusion:', confusion)

confusion_accuracy = (confusion[0][0] + confusion[1][1])/(confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1])

error_rate = (confusion[0][1] + confusion[1][0])/(confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1])

print('confusion error rate:', error_rate)
print('confusion accuracy:', confusion_accuracy)


print('Vince: ', vince_pred)


# print(xtest)

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred)) 

import seaborn as sns
sns.regplot(x='balance', y='default', data=y_pred, logistic=True)

# X = np.c_[np.ones((X.shape[0], 1)), X]
# y = y[:, np.newaxis]
# theta = np.zeros((X.shape[1], 1))

# def sigmoid(x):
#     # Activation function used to map any real value between 0 and 1
#     return 1 / (1 + np.exp(-x))

# def net_input(theta, x):
#     # Computes the weighted sum of inputs
#     return np.dot(x, theta)

# def probability(theta, x):
#     # Returns the probability after passing through sigmoid
#     return sigmoid(net_input(theta, x))

# def cost_function(self, theta, x, y):
#     # Computes the cost function for all the training samples
#     m = x.shape[0]
#     total_cost = -(1 / m) * np.sum(
#         y * np.log(probability(theta, x)) + (1 - y) * np.log(
#             1 - probability(theta, x)))
#     return total_cost

# def gradient(self, theta, x, y):
#     # Computes the gradient of the cost function at the point theta
#     m = x.shape[0]
#     return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

# def fit(self, x, y, theta):
#     opt_weights = fmin_tnc(func=cost_function, x0=theta,
#                   fprime=gradient,args=(x, y.flatten()))
#     return opt_weights[0]


# parameters = fit(X, y, theta)
