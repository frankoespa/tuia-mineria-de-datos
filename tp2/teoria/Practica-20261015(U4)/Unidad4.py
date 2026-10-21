# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 17:54:17 2023

@author: flavio
"""
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from graphviz import Source

""" Decision Tree - Classification """

target_names = {
    1:'Kama',
    2:'Rosa', 
    3:'Canadian'
}

dataWheat = pd.read_csv("wheat.csv")
dataWheat['category'] = dataWheat['category'].map(target_names)

xWheat = dataWheat.drop('category', axis=1)
yWheat = dataWheat['category']

xWheatTrain, xWheatTest, yWheatTrain, yWheatReal = train_test_split(xWheat, yWheat, test_size=0.2)

tree_clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
tree_clf.fit(xWheatTrain, yWheatTrain)

export_graphviz(tree_clf, out_file="wheat.dot", feature_names=xWheat.columns,
class_names=yWheat, rounded=True, filled=True)
Source.from_file("wheat.dot")

yWheatPred = tree_clf.predict(xWheatTest)
cm = metrics.confusion_matrix(yWheatReal, yWheatPred)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()

metrics.precision_score(yWheatReal, yWheatPred, average='macro')
metrics.recall_score(yWheatReal, yWheatPred, average='macro')
metrics.accuracy_score(yWheatReal, yWheatPred)

tree_clf.predict_proba(xWheatTest[0:4]).round(3)

pca_pipeline = make_pipeline(StandardScaler(), PCA())
xWheatRotated = pca_pipeline.fit_transform(xWheatTrain)
tree_clf_pca = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
tree_clf_pca.fit(xWheatTrain, yWheatTrain)

export_graphviz(tree_clf_pca, out_file="wheat.dot", feature_names=xWheat.columns,
class_names=yWheat, rounded=True, filled=True)
Source.from_file("wheat.dot")

yWheatPred = tree_clf_pca.predict(xWheatTest)
cm = metrics.confusion_matrix(yWheatReal, yWheatPred)
metrics.ConfusionMatrixDisplay(confusion_matrix=cm).plot()
metrics.accuracy_score(yWheatReal, yWheatPred)

""" Decision Tree - Regression """

dataDiabetes = pd.read_csv("diabetes.csv")

xDiabetes = dataDiabetes.drop('value', axis=1)
yDiabetes = dataDiabetes['value']

xDiabetesTrain, xDiabetesTest, yDiabetesTrain, yDiabetestReal = train_test_split(xDiabetes, yDiabetes, test_size=0.2)

tree_reg = DecisionTreeRegressor(max_depth=3, criterion='squared_error', random_state=13, 
                                 min_samples_leaf=1, min_samples_split=2)

tree_reg.fit(xDiabetesTrain, yDiabetesTrain)

yDiabetesPred = tree_reg.predict(xDiabetesTest)

print('Mean Absolute Error:', metrics.mean_absolute_error(yDiabetestReal, yDiabetesPred))
print('Mean Squared Error:', metrics.mean_squared_error(yDiabetestReal, yDiabetesPred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yDiabetestReal, yDiabetesPred)))

export_graphviz(tree_reg, out_file="diabetes.dot", feature_names=xDiabetes.columns,
class_names=yDiabetes, rounded=True, filled=True)
Source.from_file("diabetes.dot")

tableResult = pd.DataFrame({'Actual':yDiabetestReal, 'Predicted':yDiabetesPred})
