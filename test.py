import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

# Read the csv files
mobile_dataset = pd.read_csv('data/train_mobile.csv')
print(mobile_dataset)

y_mobile = mobile_dataset['price_range']
print(y_mobile.head())
x_mobile = mobile_dataset.drop(['price_range'], axis=1)
print(x_mobile.head())

# Set data for train and test from the Train data
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30)


class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.classes = None  # List of unique classes in training set
        self.class_prior = None  # Prior probability of each class
        self.mean = None  # Mean of each feature for each class
        self.var = None  # Variance of each feature for each class
        self.poly = None  # PolynomialFeatures object

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_prior = np.zeros(len(self.classes))
        self.mean = np.zeros((len(self.classes), X.shape[1]))
        self.var = np.zeros((len(self.classes), X.shape[1]))

        # Calculate prior probability of each class
        for i, c in enumerate(self.classes):
            self.class_prior[i] = (np.sum(y == c) + self.alpha) / (len(y) + self.alpha * len(self.classes))

            # Calculate mean and variance of each feature for each class
            X_c = X[y == c]
            self.mean[i] = np.mean(X_c, axis=0)
            self.var[i] = np.var(X_c, axis=0)

        # Fit polynomial features on the dataset
        self.poly = PolynomialFeatures(include_bias=False)
        X_poly = self.poly.fit_transform(X)
        self.mean = self.poly.transform(self.mean)
        self.var = self.poly.transform(self.var)

    def predict(self, X):
        # Transform input dataset using polynomial features
        X_poly = self.poly.transform(X)

        # Calculate log-likelihood for each class and add prior probability
        log_likelihood = np.zeros((X_poly.shape[0], len(self.classes)))
        for i, c in enumerate(self.classes):
            log_likelihood[:, i] += np.sum(norm.logpdf(X_poly, self.mean[i], np.sqrt(self.var[i])), axis=1)
            log_likelihood[:, i] += np.log(self.class_prior[i])

        # Return class with highest log-likelihood for each data point
        return self.classes[np.argmax(log_likelihood, axis=1)]


def k_fold_cv(classifier, X, y):
    kf = KFold(n_splits=10)
    accuracies = []
    for train_idx, test_idx in kf.split(X):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)


def f1_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)


nb = NaiveBayesClassifier()
k_fold_cv(nb, x_mobile, y_mobile)

y_pred = nb.predict(x_mobile_test)

y_pred = nb.predict(x_mobile_test)
f1 = f1_score(y_mobile_test, y_pred)
acc = accuracy(y_mobile_test, y_pred)

print("Accuracy= ", acc)
print("F1_score= ", f1)

