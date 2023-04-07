import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from scipy.stats import multinomial
from sklearn.model_selection import train_test_split

# Read the csv files
mobile_dataset = pd.read_csv('data/train_mobile.csv')

continuous_indices = [2, 7]  # indices of continuous features
x_mobile = mobile_dataset.iloc[:, :-1].values
y_mobile = mobile_dataset.iloc[:, 20].values

# Set data for train and test from the Train data
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30)


class NaiveBayesClassifier:
    """
    Bayes Theorem form
    P(y|X) = P(X|y) * P(y) / P(X)
    """

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
        """
        calculate probability from gaussian density function (normally distributed)
        we will assume that probability of specific target value given specific class is normally distributed

        probability density function derived from wikipedia:
        (1/√2pi*σ) * exp((-1/2)*((x-μ)^2)/(2*σ²)), where μ is mean, σ² is variance, σ is quare root of variance (standard deviation)
        """
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

    def accuracy(self, y_test, y_pred):
        accuracy = np.sum(y_test == y_pred) / len(y_test)
        return accuracy

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


# train the model
nbc = NaiveBayesClassifier()

nbc.fit(x_mobile_train, y_mobile_train)

predict_train = nbc.predict(x_mobile_test)

acc = nbc.accuracy(y_mobile_test, predict_train)
f1score = f1_score(predict_train, y_mobile_test, average='weighted')

print("\n")
print("TRAIN Naive Bayes classifier")
print("Accuracy= ", acc)
print("f1 score(weighted)= ", f1score)
