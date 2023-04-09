import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

mobile_dataset = pd.read_csv("data/train_mobile.csv")

y_mobile = mobile_dataset['price_range']
print(y_mobile.head())
x_mobile = mobile_dataset.drop(['price_range'], axis=1)
print(x_mobile.head())

x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=1)

""""" 1 hidden layer Neural Network """""
# Initializing the MLPClassifier
k = 200
classifier = MLPClassifier(hidden_layer_sizes=k, max_iter=800, activation='tanh', solver='sgd', alpha=0.0001,
                           random_state=1)

# Fitting the training data to the network
classifier.fit(x_mobile_train, y_mobile_train)

# Predicting y for x_test
y_pred = classifier.predict(x_mobile_test)

# Accuracy and f1 score for Train data
acc = accuracy_score(y_pred, y_mobile_test)
score = f1_score(y_pred, y_mobile_test, average='weighted')

print("\n1 Hidden Layer NN")
print("Accuracy= ", acc)
print("F1 score= ", score)


""""" 2 hidden layer Neural Network """""
# Initializing the MLPClassifier
k1 = 200
k2 = 100
classifierB = MLPClassifier(hidden_layer_sizes=(k1, k2), max_iter=800, activation='tanh', solver='sgd', random_state=1)

# Fitting the training data to the network
classifierB.fit(x_mobile_train, y_mobile_train)

# Predicting y for x_test
y_predB = classifierB.predict(x_mobile_test)

# Accuracy and f1 score for Train data
accB = accuracy_score(y_predB, y_mobile_test)
scoreB = f1_score(y_predB, y_mobile_test, average='weighted')

print("\n2 Hidden Layer NN")
print("Accuracy= ", accB)
print("F1 score= ", scoreB)
