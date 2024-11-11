import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
print(os.getcwd())
file_path = r"C:\Users\user\Desktop\cancer_detection\uploads\data.csv"
df = pd.read_csv(file_path)
df = df.dropna(axis = 1)
# print(df.shape)
y = df['diagnosis']
x = df.drop(columns=["diagnosis", "id"], axis=1)
print(x.shape)

#Train test split()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# random state to ensure random shuffling of data
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
inst = DecisionTreeClassifier()
# An instance of decision tree classifier which represents our decision tree model
parameters = {"max_depth":[1, 2, 3, 4, 5, 7, 10],
              "min_samples_leaf": [1, 3, 6, 10, 20]}
# parameter we want to tune our model with
#  max_depth = maximum depth of the tree
# min_samples leaf  = minimum no of sample required to be in a leaf node

clf = GridSearchCV(inst, parameters, n_jobs= -1)
clf.fit(x_train,y_train)
# n_jobs =  no of CPU cores
# Grid search test different combinations of hyperparameters
print(clf.best_params_)
# print the best parameters
model = clf.fit(x_train, y_train)
# 1 denotes malignat
# 0 denotes benign
prediction = model.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediction))
print(prediction)
features = [13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259]
prediction1 = model.predict([features])[0]
print(type(prediction1))
result = 'Malignant' if prediction1 == 'M' else 'Benign'
print(result)
# prediction = model.predict(x_test)
# print(prediction[0])
# m_count,b_count = 0,0
# for pred in prediction:
#     if pred == 'M':
#         print("Result: Malignant")
#         m_count += 1

#     else:
#         print("Result: Benign")
#         b_count += 1
# print("Total Malignant: ", m_count)
# print("Total Benign: ", b_count)
