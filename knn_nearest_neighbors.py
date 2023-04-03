import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.distance import hamming
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import DistanceMetric

data = "data/train_mobile.csv"

labels = ['battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 'four_g', 'int_memory', 'm_dep', 'mobile_wt',
          'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'three_g', 'touch_screen',
          'wifi', 'price_range']

# load data from csv
dataset = pd.read_csv(data)
print(dataset.head())

continuous_indices = [2, 7]  # indices of continuous features
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 20].values

# split data to train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


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


classifier = KNeighborsClassifier(n_neighbors=3, metric=custom_distance,
                                  metric_params={'continuous_indices': continuous_indices})
classifier.fit(x_train, y_train)

y_predict = classifier.predict(x_test)

result = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_predict)
print("Classification Report:")
print(result1)
result2 = accuracy_score(y_test, y_predict)
print("Accuracy:", result2)

np.savetxt('output_KNN.csv', np.array(y_predict).astype(int), delimiter=',', fmt='%d')
