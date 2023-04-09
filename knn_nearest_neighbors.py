import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import hamming
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


########## Mobile Price Classification ##########
"""
# load data from csv
mobile_data = "data/train_mobile.csv"
mobile_dataset = pd.read_csv(mobile_data)
print(mobile_dataset.head())

continuous_indices = [2, 7]  # indices of continuous features
y_mobile = mobile_dataset['price_range']
x_mobile = mobile_dataset.drop(['price_range'], axis=1)

# split data to train and test sets
x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30)

scaler = StandardScaler()
scaler.fit(x_mobile_train)
x_mobile_train = scaler.transform(x_mobile_train)
x_mobile_test = scaler.transform(x_mobile_test)


# K-NN Classifier
def custom_distance(x1, x2, continuous_indices):
    # Euclidean distance for continuous features
    euclidean_distance = np.linalg.norm(x1[continuous_indices] - x2[continuous_indices])

    # Hamming distance for discrete features
    hamming_distance = hamming(x1[np.where(~np.array(continuous_indices))[0]],
                               x2[np.where(~np.array(continuous_indices))[0]])

    # Total distance
    total_distance = euclidean_distance + hamming_distance

    return total_distance


mobile_classifier = KNeighborsClassifier(n_neighbors=3, metric=custom_distance,
                                         metric_params={'continuous_indices': continuous_indices})
mobile_classifier.fit(x_mobile_train, y_mobile_train)

y_mobile_predict = mobile_classifier.predict(x_mobile_test)

result = confusion_matrix(y_mobile_test, y_mobile_predict)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_mobile_test, y_mobile_predict)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_mobile_test, y_mobile_predict)
print("Accuracy:", result2)

np.savetxt('mobile_output_KNN.csv', np.array(y_mobile_predict).astype(int), delimiter=',', fmt='%d')
"""

########## Airlines Delay ##########

airlines_data = 'data/airlines_delay.csv'
airlines_dataset = pd.read_csv(airlines_data)
airlines_dataset.drop("Flight", axis=1, inplace=True)
print(airlines_dataset)

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

print(airlines_dataset)

continuous_indices = [0, 1]  # indices of continuous features
x_airlines = airlines_dataset.iloc[:, :-1].values
y_airlines = airlines_dataset.iloc[:, 6].values

# split data to train and test sets
x_airlines_train, x_airlines_test, y_airlines_train, y_airlines_test = train_test_split(x_airlines, y_airlines,
                                                                                        test_size=0.30)

scaler = StandardScaler()
scaler.fit(x_airlines_train)
x_airlines_train = scaler.transform(x_airlines_train)
x_airlines_test = scaler.transform(x_airlines_test)


# K-NN Classifier
def distance(x1, x2, continuous_indices):
    # Euclidean distance for continuous features
    euclidean_distance = np.linalg.norm(x1[continuous_indices] - x2[continuous_indices])

    # Hamming distance for discrete features
    hamming_distance = hamming(x1[np.where(~np.array(continuous_indices))[0]],
                               x2[np.where(~np.array(continuous_indices))[0]])

    # Total distance
    total_distance = euclidean_distance + hamming_distance

    return total_distance


airlines_classifier = KNeighborsClassifier(n_neighbors=3)
airlines_classifier.fit(x_airlines_train, y_airlines_train)

y_airline_predict = airlines_classifier.predict(x_airlines_test)

result_airlines = confusion_matrix(y_airlines_test, y_airline_predict)
print("Confusion Matrix:")
print(result_airlines)
result1_airlines = classification_report(y_airlines_test, y_airline_predict)
print("Classification Report:")
print(result1_airlines)
result2_airlines = accuracy_score(y_airlines_test, y_airline_predict)
print("Accuracy:", result2_airlines)
