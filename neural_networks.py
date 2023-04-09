import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

""" Mobile Dataset """
mobile_dataset = pd.read_csv("data/train_mobile.csv")
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)
neurons = [50, 100, 200]
n_pairs = [(50, 25), (100, 50), (200, 100)]

x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=1)

""""" 1 hidden layer Neural Network """""
print("\n1 Hidden Layers NN")
# Initializing the MLPClassifier
for k in neurons:
    classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd', random_state=1)

    # Fitting the training data to the network
    classifier.fit(x_mobile_train, y_mobile_train)

    # Predicting y for x_test
    y_pred = classifier.predict(x_mobile_test)

    # Accuracy and f1 score for Train data
    acc = accuracy_score(y_pred, y_mobile_test)
    score = f1_score(y_pred, y_mobile_test, average='weighted')

    print(k, "hidden neurons")
    print("Accuracy= ", acc)
    print("F1 score= ", score, "\n")


""""" 2 hidden layers Neural Network """""
print("\n2 Hidden Layers NN")
# Initializing the MLPClassifier
for k in n_pairs:
    two_hidden_layers_classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd', random_state=1)

    # Fitting the training data to the network
    two_hidden_layers_classifier.fit(x_mobile_train, y_mobile_train)

    # Predicting y for x_test
    y_pred = two_hidden_layers_classifier.predict(x_mobile_test)

    # Accuracy and f1 score for Train data
    acc = accuracy_score(y_pred, y_mobile_test)
    score = f1_score(y_pred, y_mobile_test, average='weighted')

    print(k, "hidden neurons")
    print("Accuracy= ", acc)
    print("F1 score= ", score, "\n")
