import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

""" Mobile Price Classification """
# load data from csv
mobile_data = "data/train_mobile.csv"
mobile_dataset = pd.read_csv(mobile_data)

# k-neighbors
neighbors = [1, 3, 5, 10]

# continuous_indices = [2, 7]  # indices of continuous features
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# split data to train and test sets
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30)

scaler = StandardScaler()

print("k-NN Mobile Dataset")
for k in neighbors:
    mobile_knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Implement 10-fold cross-validation on the training set
    scores = cross_val_score(mobile_knn_classifier, scaler.fit_transform(x_mobile), y_mobile, cv=10)

    # Fit the classifier to the training data for each fold of the cross-validation
    for train_index, test_index in KFold(n_splits=10).split(x_mobile):
        x_mobile_train, x_mobile_test = x_mobile.iloc[train_index], x_mobile.iloc[test_index]
        y_mobile_train, y_mobile_test = y_mobile.iloc[train_index], y_mobile.iloc[test_index]
        mobile_knn_classifier.fit(scaler.fit_transform(x_mobile_train), y_mobile_train)
        y_mobile_predict = mobile_knn_classifier.predict(scaler.fit_transform(x_mobile_test))

        print("k-NN (k="+str(k)+")")
        mobile_acc = accuracy_score(y_mobile_predict, y_mobile_test)
        print("Accuracy:", mobile_acc)
        f1score = f1_score(y_mobile_predict, y_mobile_test, average='weighted')
        print("f1 score: ", f1score)


""" Airlines Delay """
airlines_data = 'data/airlines_delay.csv'
airlines_dataset = pd.read_csv(airlines_data)
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
scaler.fit(x_airlines_train)
x_airlines_train = scaler.transform(x_airlines_train)
x_airlines_test = scaler.transform(x_airlines_test)

print("\nk-NN Airlines Dataset")
for k in neighbors:
    airlines_knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Implement 10-fold cross-validation on the training set
    scores = cross_val_score(airlines_knn_classifier, scaler.fit_transform(x_airlines), y_airlines, cv=10)

    # Fit the classifier to the training data for each fold of the cross-validation
    for train_index, test_index in KFold(n_splits=10).split(x_airlines):
        x_airlines_train, x_airlines_test = x_airlines.iloc[train_index], x_airlines.iloc[test_index]
        y_airlines_train, y_airlines_test = y_airlines.iloc[train_index], y_airlines.iloc[test_index]
        airlines_knn_classifier.fit(scaler.fit_transform(x_airlines_train), y_airlines_train)
        y_airlines_predict = airlines_knn_classifier.predict(scaler.fit_transform(x_airlines_test))

        print("k-NN (k="+str(k)+")")
        airlines_acc = accuracy_score(y_airlines_predict, y_airlines_test)
        print("Accuracy:", airlines_acc)
        f1score = f1_score(y_airlines_predict, y_airlines_test, average='weighted')
        print("f1 score: ", f1score)
