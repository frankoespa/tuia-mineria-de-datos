# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:26:33 2023

@author: Flavio E Spetale
"""
# Data Processing
import pandas as pd

# Modelling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint

# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import display
import graphviz

dataWheat = pd.read_csv("wheat.csv")

xWheat = dataWheat.drop('category', axis=1)
yWheat = dataWheat['category']

xWheatTrain, xWheatTest, yWheatTrain, yWheatReal = train_test_split(xWheat, yWheat, test_size=0.2)

rf = RandomForestClassifier()
rf.fit(xWheatTrain, yWheatTrain)

y_pred = rf.predict(xWheatTest)

accuracy = accuracy_score(yWheatReal, y_pred)
print("Accuracy:", accuracy)

# Export the first three decision trees from the forest
for i in range(3):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=xWheatTrain.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)
    display(graph)
    
param_dist = {'n_estimators': randint(50,500), 'max_depth': randint(1,20)}  
    
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, param_distributions = param_dist, 
                                 n_iter=5, cv=5)

# Fit the random search object to the data
rand_search.fit(xWheatTrain, yWheatTrain)

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Generate predictions with the best model
yWheatPred = best_rf.predict(xWheatTest)

# Create the confusion matrix
cm = confusion_matrix(yWheatReal, yWheatPred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

precision_score(yWheatReal, yWheatPred, average='macro')
recall_score(yWheatReal, yWheatPred, average='macro')
accuracy_score(yWheatReal, yWheatPred)