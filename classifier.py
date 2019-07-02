import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import csv
from pprint import pprint


dataset = pd.read_csv('data.csv')
dataset = dataset.iloc[:, :].values
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(dataset[:, 2:17])
dataset[:, 2:17] = imputer.transform(dataset[:, 2:17])
# pprint(dataset)

dataset = pd.DataFrame(dataset[:, 2:])
# pprint(dataset)
X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
y = y.astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/10, random_state=1)

# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm='SAMME', n_estimators=10).fit(X_train, y_train)
# classifier = RandomForestClassifier().fit(X_train, y_train)
# classifier = MLPClassifier().fit(X_train, y_train)
# classifier = GaussianNB().fit(X_train, y_train)

acc = classifier.score(X_test, y_test)
print(acc)
