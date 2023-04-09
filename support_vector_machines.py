from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

mobile_dataset = pd.read_csv("data/train_mobile.csv")

y_mobile = mobile_dataset['price_range']
print(y_mobile.head())
x_mobile = mobile_dataset.drop(['price_range'], axis=1)
print(x_mobile.head())

x_mobile_train, x_mobile_test, y_mobile_train, y_mobile_test = train_test_split(x_mobile, y_mobile, test_size=0.30, random_state=1)

""" Linear SVM """
svm_linear = LinearSVC()

svm_linear.fit(x_mobile_train,y_mobile_train)
y_pred_linear = svm_linear.predict(x_mobile_test)

# Accuracy and f1 score for Train data
acc_linear = accuracy_score(y_pred_linear, y_mobile_test)
score_linear = f1_score(y_pred_linear, y_mobile_test, average='weighted')

print("\n")
print("Linear SVM")
print("Accuracy= ", acc_linear)
print("f1 score= ", score_linear)


""" Gaussian SVM """

C_2d_range = [1e-2, 1, 1e2]
gamma_2d_range = [1e-1, 1, 1e1]
scores = []
for c in C_2d_range:
    for gamma in gamma_2d_range:
        svm_gaussian = SVC(C=c, gamma=gamma)
        svm_gaussian.fit(x_mobile_train, y_mobile_train)
        y_pred_gaussian = svm_gaussian.predict(x_mobile_test)
        # Accuracy and f1 score for data
        acc_gaussian = accuracy_score(y_pred_gaussian, y_mobile_test)
        score_gaussian = f1_score(y_pred_gaussian, y_mobile_test, average='weighted')
        scores.append((acc_gaussian, score_gaussian))


print(scores)
best_scores = max(scores)

print("\n")
print("Gaussian SVM")
print("Accuracy= ", best_scores[0])
print("f1 score= ", best_scores[1])
