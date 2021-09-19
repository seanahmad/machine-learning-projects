# Iris Classification Model

 In this repository, I'll go through a simple machine learning application and create my first model.


My goal is to build a machine learning model that can learn from the measurements of these irises whose species is known, so that we can predict the species for a new iris (setosa, versicolor, or virginica).

# Run Model Locally 

1. Open your Terminal enter:
``` bash
    git clone https://github.com/gaurtvin/iris-classification-model.git
```

2. Do to Code Directory
``` bash
    cd model
```
3. Run the file in Terminal
``` bash
    python iris_code.py
```


## Here is a summary of the Code needed for the whole training and Evaluation procedure:

```python
import sys
import pandas as pd
import matplotlib
import numpy as np
import scipy as sp
import IPython
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

# loading dataset
iris_dataset = load_iris()

# exploring data
print("Key of iris_datasets: \n {}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target name: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("Shape of data: {}".format(iris_dataset['data'].shape))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))


print("Type of target: {}".format(type(iris_dataset['target'])))
print("Shape of target: {}".format(iris_dataset['target'].shape))
print("Target:\n{}".format(iris_dataset['target']))

# Split into Training and Testing Data
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))

print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# Implementing KNN k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making Predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new.shape: {}".format(X_new.shape))

prediction = knn.predict(X_new)
print("Prediction: {}".format(prediction))
print("Predicted target name: {}".format(
    iris_dataset['target_names'][prediction]))

# Evaluation
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# we can also use
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# 97%


# Summary

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

```

The **Iris dataset** consists of two NumPy arrays: one containing the data, which is
referred to as X in scikit-learn, and one containing the correct or desired outputs,
which is called y. The array X is a two-dimensional array of features, with one row per
data point and one column per feature. The array y is a one-dimensional array, which
here contains one class label, an integer ranging from 0 to 2, for each of the samples.
We split our dataset into a training set, to build our model, and a test set, to evaluate
how well our model will generalize to new, previously unseen data.

### We chose the **K-nearest neighbors classification algorithm**, which makes predictions.

For a new data point by considering its closest neighbor(s) in the training set. This is
implemented in the KNeighborsClassifier class, which contains the algorithm that
builds the model as well as the algorithm that makes a prediction using the model.
We instantiated the class, setting parameters. Then we built the model by calling the
fit method, passing the training data (X_train) and training outputs (y_train) as
parameters. We evaluated the model using the score method, which computes the
accuracy of the model. We applied the score method to the test set data and the test
set labels and found that our model is about 97% accurate, meaning it is correct 97%
of the time on the test set.
This gave us the confidence to apply the model to new data (in our example, new
flower measurements) and trust that the model will be correct about **97%** of the time.

# Contribute
### Clone this repository

1. Open your Terminal
``` bash
    git clone https://github.com/gaurtvin/iris-classification-model.git
```

2. Create new Branch
``` bash
    git branch your-name
```
### Create Pull Request ...

# License
### This repository is licensed Under [MIT](/License) Lisence.

