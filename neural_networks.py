import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder


# MLPClassifier class
# MLPClassifier class
class MLP:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    # 1 Hidden Layer MLP Classifier
    def MLPClass(self):
        print("\n1 Hidden Layers NN")
        neurons = [50, 100, 200]
        for k in neurons:
            classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd',
                                       random_state=1)

            # Cross-validation
            scores = cross_val_score(classifier, self.x_train, self.y_train, cv=10)
            print("Cross-validation scores for", k, "hidden neurons:", scores)
            print("Mean accuracy:", np.mean(scores))

            # Fitting the training data to the network
            classifier.fit(self.x_train, self.y_train)

            # Predicting y for x_test
            y_pred = classifier.predict(self.x_test)

            # Accuracy and f1 score for Test data
            acc = accuracy_score(y_pred, self.y_test)
            f1score = f1_score(y_pred, self.y_test, average='weighted')

            print(k, "hidden neurons")
            print("Accuracy= ", acc)
            print("F1 score= ", f1score, "\n")

    # 2 Hidden Layer MLP Classifier
    def MLPClass2(self):
        print("\n2 Hidden Layers NN")
        n_pairs = [(50, 25), (100, 50), (200, 100)]
        for k in n_pairs:
            two_hidden_layers_classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh',
                                                         solver='sgd',
                                                         random_state=1)

            # Cross-validation
            scores = cross_val_score(two_hidden_layers_classifier, self.x_train, self.y_train, cv=10)
            print("Cross-validation scores for", k, "hidden neurons:", scores)
            print("Mean accuracy:", np.mean(scores))

            # Fitting the training data to the network
            two_hidden_layers_classifier.fit(self.x_train, self.y_train)

            # Predicting y for x_test
            y_pred = two_hidden_layers_classifier.predict(self.x_test)

            # Accuracy and f1 score for Test data
            acc = accuracy_score(y_pred, self.y_test)
            f1score = f1_score(y_pred, self.y_test, average='weighted')

            print(k, "hidden neurons")
            print("Accuracy= ", acc)
            print("F1 score= ", f1score, "\n")


""" Mobile Dataset """
mobile_dataset = pd.read_csv("data/train_mobile.csv")
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30,
                                                                                random_state=1)

print("Mobile Price Dataset")
mlp_mobile = MLP(x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test)
mlp_mobile.MLPClass()
mlp_mobile.MLPClass2()

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

print("Airlines Delay Dataset")
mlp_airlines = MLP(x_airlines_train, x_airlines_test, y_airlines_train, y_airlines_test)
mlp_airlines.MLPClass()
mlp_airlines.MLPClass2()
