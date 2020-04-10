import sklearn
import sklearn.linear_model
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

FEATURES = [
    'G1',
    'G2',
    'G3',
    'studytime',
    'failures',
    'romantic',
    'freetime',
    'goout',
    'health',
    'school'
]

studentData = pd.read_csv('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/StudentPerformance/studentData.csv', sep=';')
studentData = studentData[FEATURES]
studentData = shuffle(studentData)

studentData.loc[studentData['romantic'] == 'yes', 'romantic'] = 1
studentData.loc[studentData['romantic'] == 'no', 'romantic'] = 0
studentData.loc[studentData['school'] == 'GP', 'school'] = 1
studentData.loc[studentData['school'] == 'MS', 'school'] = 0

predict = 'G3'

x = np.array(studentData.drop([predict], axis=1))
y = np.array(studentData[predict])

# linear = sklearn.linear_model.LinearRegression()

# best = 0
# for _ in range(50):
#     xTest, xTrain, yTest, yTrain = sklearn.model_selection.train_test_split(x, y, test_size=0.1, shuffle=True)
#     linear.fit(xTrain, yTrain)
#     accuracy = linear.score(xTest, yTest)
#     if accuracy > best:
#         best = accuracy
#         with open('trainedModel.pickle', 'wb') as f:
#             pickle.dump(linear, f)

xTest, xTrain, yTest, yTrain = sklearn.model_selection.train_test_split(x, y, test_size=0.1, shuffle=True)

pickleIn = open('trainedModel.pickle', 'rb')
linear = pickle.load(pickleIn)
best = linear.score(xTrain, yTrain)

print('Accuracy : ', best*100)
print('Intercept : ', linear.intercept_)
print('Coefficient : ', linear.coef_)

predictions = linear.predict(xTest)

p = 'freetime'
plt.scatter(studentData[p], studentData['G3'], c=y)
plt.xlabel(p)
plt.ylabel('G3')
plt.show()