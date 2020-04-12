import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing, linear_model, neighbors

carsData = pd.read_csv('/home/aryan/Desktop/Machine Learning (Coursera)/Python Practice/Cars/car.data', sep=',')

L = preprocessing.LabelEncoder()

buying = L.fit_transform(list(carsData['buying']))
maint = L.fit_transform(list(carsData['maint']))
door = L.fit_transform(list(carsData['door']))
persons = L.fit_transform(list(carsData['persons']))
lug_boot = L.fit_transform(list(carsData['lug_boot']))
safety = L.fit_transform(list(carsData['safety']))
cls = L.fit_transform(list(carsData['class']))

x = list(zip(buying,maint,door,persons,lug_boot,safety))
y = list(cls)

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.1, shuffle=True)

model = neighbors.KNeighborsClassifier(n_neighbors=7)
model.fit(xTrain, yTrain)
accuracy = model.score(xTest, yTest)

print('Accuracy : ', accuracy*100, '%')

names = ['unacc', 'acc', 'good', 'vgood']

predicted = model.predict(xTest)
for i in range(len(xTest)):
    if predicted[i] != yTest[i]:
        print('Data : ', xTest[i], 'Predicted : ',names[predicted[i]], 'Actual Value : ', names[yTest[i]])

n = model.kneighbors(xTest, 3, True)
for i in range(len(n[0])):
    print('Distance : ', n[0][i], 'Indices : ', n[1][i])