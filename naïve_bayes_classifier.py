import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


class NaiveBayesClassifier:
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

    def visualize(self, y_true, y_pred, target):

        tr = pd.DataFrame(data=y_true, columns=[target])
        pr = pd.DataFrame(data=y_pred, columns=[target])

        fig, ax = plt.subplots(1, 2, sharex='col', sharey='row', figsize=(15, 6))

        sns.countplot(x=target, data=tr, ax=ax[0], palette='viridis', alpha=0.7, hue=target, dodge=False)
        sns.countplot(x=target, data=pr, ax=ax[1], palette='viridis', alpha=0.7, hue=target, dodge=False)

        fig.suptitle('True vs Predicted Comparison', fontsize=20)

        ax[0].tick_params(labelsize=12)
        ax[1].tick_params(labelsize=12)
        ax[0].set_title("True values", fontsize=18)
        ax[1].set_title("Predicted values", fontsize=18)
        plt.show()


""" Mobile Price """
# Read the csv files
mobile_dataset = pd.read_csv('data/train_mobile.csv')
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# Set data for train and test from the Train data
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30)


# Define the KFold object
kfold = KFold(n_splits=10)

# Initialize the arrays for storing accuracy and f1 score
scores = []

# Iterate over the splits and train/test the model
for train_idx, test_idx in kfold.split(x_mobile, y_mobile):
    x_train, y_train = x_mobile.iloc[train_idx], y_mobile.iloc[train_idx]
    x_test, y_test = x_mobile.iloc[test_idx], y_mobile.iloc[test_idx]

    # train the model
    naive_bayes_classifier = NaiveBayesClassifier()
    naive_bayes_classifier.fit(x_train, y_train)

    predict_test = naive_bayes_classifier.predict(x_test)

    acc = accuracy_score(y_test, predict_test)
    f1score = f1_score(predict_test, y_test, average='weighted')

    scores.append((acc,f1score))

print("\nMobile Price dataset\n")
s = max(scores)
print("Naive Bayes classifier best score of 10-fold cross-validation")
print("Accuracy: ", s[0])
print("f1 score: ", s[1])


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

# continuous_indices = [0, 1]  # indices of continuous features
x_airlines = airlines_dataset.drop(['Class'], axis=1)
y_airlines = airlines_dataset['Class']

# split data to train and test sets
x_airlines_train, x_airlines_test, y_airlines_train, y_airlines_test = train_test_split(x_airlines, y_airlines,
                                                                                        test_size=0.30)

scaler = StandardScaler()

# Define the KFold object
kfold = KFold(n_splits=10)

# Initialize the arrays for storing accuracy and f1 score
scores = []

# Iterate over the splits and train/test the model
for train_idx, test_idx in kfold.split(x_airlines, y_airlines):
    x_train, y_train = x_airlines.iloc[train_idx], y_airlines.iloc[train_idx]
    x_test, y_test = x_airlines.iloc[test_idx], y_airlines.iloc[test_idx]

    # train the model
    naive_bayes_classifier = NaiveBayesClassifier()
    naive_bayes_classifier.fit(x_train, y_train)

    predict_test = naive_bayes_classifier.predict(x_test)

    acc = accuracy_score(y_test, predict_test)
    f1score = f1_score(predict_test, y_test, average='weighted')

    scores.append((acc, f1score))

print("\nAirlines Delay dataset\n")
s = max(scores)
print("Naive Bayes classifier best score of 10-fold cross-validation")
print("Accuracy: ", s[0])
print("f1 score: ", s[1])
