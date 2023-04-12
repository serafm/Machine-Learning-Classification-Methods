import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


class NaiveBayesClassifier:
    def __init__(self, n_folds=10):
        self.n_folds = n_folds
        self.classes = None
        self.count = None
        self.feature_nums = None
        self.rows = None
        self.mean = None
        self.var = None
        self.prior = None

    def calc_prior(self, features, target):
        """
        prior probability P(y)
        calculate prior probabilities
        """
        self.prior = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy()

        return self.prior

    def calc_statistics(self, features, target):
        """
        calculate mean, variance for each column and convert to numpy array
        """
        self.mean = features.groupby(target).apply(np.mean).to_numpy()
        self.var = features.groupby(target).apply(np.var).to_numpy()

        return self.mean, self.var

    def gaussian_density(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        prob = numerator / denominator
        return prob

    def multinomial_distribution(self, x):
        md = multinomial(6, [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        p = md.pmf(x)

    def calc_posterior(self, x):
        posteriors = []

        # calculate posterior probability for each class
        for i in range(self.count):
            prior = np.log(self.prior[i])  # use the log to make it more numerically stable

            if i < 4:
                conditional = np.sum(
                    np.log(self.gaussian_density(i, x)))  # use the log to make it more numerically stable
            else:
                conditional = self.multinomial_distribution(x.iloc[:, 4])

            posterior = prior + conditional
            posteriors.append(posterior)

        # return class with the highest posterior probability
        return self.classes[np.argmax(posteriors)]

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.feature_nums = features.shape[1]
        self.rows = features.shape[0]

        self.calc_statistics(features, target)
        self.calc_prior(features, target)

    def predict(self, features):
        preds = [self.calc_posterior(f) for f in features.to_numpy()]
        return preds


def classifier_train(X, y, x_train, x_test, y_train, y_test):
    # Define the KFold object
    kfold = KFold(n_splits=10)

    # Initialize the arrays for storing accuracy and f1 score
    acc_scores = []
    f1_scores = []

    naive_bayes_classifier = NaiveBayesClassifier()
    naive_bayes_classifier.fit(x_train, y_train)

    # evaluate classifier on test data
    test_preds = naive_bayes_classifier.predict(x_test)
    test_acc = accuracy_score(test_preds, y_test)
    test_f1 = f1_score(test_preds, y_test, average='weighted')

    # Iterate over the splits and train/test the model
    for train_idx, test_idx in kfold.split(X, y):
        x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        x_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # train the model
        naive_bayes_classifier = NaiveBayesClassifier()
        naive_bayes_classifier.fit(x_train, y_train)

        predict_test = naive_bayes_classifier.predict(x_test)

        acc = accuracy_score(y_test, predict_test)
        f1score = f1_score(predict_test, y_test, average='weighted')

        acc_scores.append(acc)
        f1_scores.append(f1score)

    print("Naive Bayes Classifier")
    print("Accuracy:", test_acc)
    print("F1 score:", test_f1)
    print("\n10-fold cross-validation")
    print("Mean Accuracy: ", sum(acc_scores) / len(acc_scores))
    print("Mean f1 score: ", sum(f1_scores) / len(f1_scores))


""" Mobile Price """
# Read the csv files
mobile_dataset = pd.read_csv('data/train_mobile.csv')
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# split data to train and test sets
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=42)

# Naive Bayes Classifier
print("Mobile Data Naive Bayes Classifier")
classifier_train(x_mobile, y_mobile, x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test)

""" Airlines Delay """
airlines_dataset = pd.read_csv('data/airlines_delay.csv')
airlines_dataset.drop("Flight", axis=1, inplace=True)

# Creating an instance of label Encoder.
encode_labels = LabelEncoder()

# Using .fit_transform function to fit label
# encoder and return encoded label
airline_label = encode_labels.fit_transform(airlines_dataset['Airline'])
airportFrom_labels = encode_labels.fit_transform(airlines_dataset['AirportFrom'])
airportTo_labels = encode_labels.fit_transform(airlines_dataset['AirportTo'])

# Appending the array to our dataFrame
# with column name 'Airline'
airlines_dataset["Airline"] = airline_label

# Appending the array to our dataFrame
# with column name 'AirportFrom'
airlines_dataset["AirportFrom"] = airportFrom_labels

# Appending the array to our dataFrame
# with column name 'AirportTo'
airlines_dataset["AirportTo"] = airportTo_labels

x_airlines = airlines_dataset.drop(['Class'], axis=1)
y_airlines = airlines_dataset['Class']

# split data to train and test sets
x_airlines_train, x_airlines_test, y_airlines_train, y_airlines_test = train_test_split(x_airlines, y_airlines, test_size=0.30, random_state=42)

# Naive Bayes Classifier
print("Airlines Data Naive Bayes Classifier")
classifier_train(x_airlines, y_airlines, x_airlines_train, x_airlines_test, y_airlines_train, y_airlines_test)
