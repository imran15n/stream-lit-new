# Importing Libraries

import streamlit as st
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# Importing models from scikit-learn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Define Main title of web app
st.title("My First Machine Learning Web app Using Streamlit")

# Writing Text
st.write("""
# Explore different classifier and datasets
Which one is the best?
""")
# Create datasets selection box on left side of the Web page
dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris ",'Breast Cancer','Wine Quality'))

# Using String format (f) to print selected dataset name
st.write(f"## {dataset_name} Dataset")

# Selection box for classifier

classifier_name = st.sidebar.selectbox("Select Classifier", ('KNN', 'SVM', 'Random Forest'))

# Create a function to Get a dataset

def get_dataset(name):
    if name == 'Iris':
        data = datasets.load_iris()
    elif name == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y

# calling det_dataset function to store result in two variables
X, y =  get_dataset(dataset_name)

# Getting shape of selecetd dataset
st.write('Shape of dataset:', X.shape)

# No of classes for classifiaction in dataset
st.write('Number of classes :', len(np.unique(y)))

st.write( 'Classes Name :', np.unique(y))

# Create a function to add parameters of classifiers/models

def add_params(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] =  max_depth

        n_estimator = st.sidebar.slider('n_estimator ',1, 10)
        params['n_estimator '] =  n_estimator
    return params

params =  add_params(classifier_name) # calling add_params function

# Creating a function to get a classifier by using its name and parameters settings

def get_classifiers(clf_name,params):
    clf = None
    if clf_name =='KNN':
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C = params['C'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimator '],
                                     max_depth =  params['max_depth'], random_state = 1234)
    return clf

clf = get_classifiers(classifier_name,params) # calling get_classifiers function

# Model Training and test

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.25,random_state=1234)


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Getting accuracy of selected model
acc = accuracy_score(y_test, y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy = ', acc)

### Visualize dataset ###
# Plot Data using PCA based on two primary Principal Components

pca = PCA(2)

X_project = pca.fit_transform(X)

x1 = X_project[:, 0]
x2 = X_project[:, 1]

fig = plt.figure()
plt.scatter(x1, x2,
            c=y, alpha=0.8,
            cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

st.pyplot(fig)
