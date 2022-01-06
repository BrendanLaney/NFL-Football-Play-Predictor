# NFL Footbal Play Predictor
# Predicts NFL football play based on certain conditions
# Brendan Laney
# Dr. Simon Foo
# EEE4773 Project

# Load libraries
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

path_to_file = "reg_pbp_2019_rev.csv"
play = pd.read_csv(path_to_file)

np.nan_to_num(play)

#split dataset in features and target variable
feature_cols=['down', 'qtr', 'score_differential','ydstogo']

X = play[feature_cols] # Features
y = play.play_type # Target variable

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))