# Machine Learning Classification Algorithms

This project contains an implementation of four classification algorithms: K-Nearest Neighbors (KNN), Naive Bayes Classifier (NBC), Neural Networks (NN), and Support Vector Machines (SVM). The aim of this project is to classify data into different categories based on certain features.

The datasets used in this project are from Kaggle, one for mobile price classification and the other for airline delay classification. The mobile price dataset contains features such as battery power, RAM, internal memory, and price range, while the airline delay dataset contains features such as airline name, origin airport, destination airport, and departure delay.

### Techniques Used
Two techniques were used to evaluate the performance of the algorithms: train_test_split and 10-fold cross validation.

In the train_test_split technique, the datasets were split into a training set and a testing set. The training set was used to train the algorithms, while the testing set was used to evaluate the performance of the algorithms.

In the 10-fold cross validation technique, the datasets were divided into 10 subsets of equal size. The algorithm was trained on 9 subsets and tested on the remaining subset. This process was repeated 10 times, with each subset being used as the test set once. The results were then averaged to obtain a final performance metric.

### Results
The performance of each algorithm was evaluated using accuracy and F1-score.