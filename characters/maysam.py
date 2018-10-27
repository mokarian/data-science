from sklearn import svm
from sklearn.metrics import accuracy_score
import csv
import numpy as np


def main():
    map = readCharacterMappingInfo()
    X_train, y_train = readCSVForXAndY(map, 'files/ds1Train.csv')
    X_test, y_test = readCSVForXAndY(map, 'files/ds1Val.csv')

    trainNaiveBayes(X_test, X_train, y_test, y_train)
    trainKNN(X_test, X_train, y_test, y_train)
    trainSVM(X_test, X_train, y_test, y_train)
    # trainLinearRegression(X_test, X_train, y_test, y_train)


def readCharacterMappingInfo():
    map = {}
    with open('ds1Info.csv') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            map[row[0]] = row[1]
    return map


def readCSVForXAndY(map, fileName):
    X_train = []
    y_train = []
    with open(fileName) as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        for row in csvReader:
            y_train.append(map[row[len(row) - 1]])
            row = row[:-1]
            X_train.append(np.array(row).astype(np.float))
    return X_train, y_train


def trainNaiveBayes(X_test, X_train, y_test, y_train):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy of the test data using Naive Bayes model is:" + str(accuracy_score(y_pred, y_test)) + "%")


def trainSVM(X_test, X_train, y_test, y_train):
    clf = svm.SVC(kernel='linear', gamma=0.01, C=100)
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy of the test data using SVM is:" + str(accuracy_score(y_pred, y_test)) + "%")


def trainKNN(X_test, X_train, y_test, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    print("Accuracy of the test data using KNN is:" + str(accuracy_score(y_pred, y_test)) + "%")


def trainLinearRegression(X_test, X_train, y_test, y_train):
    from sklearn import linear_model
    from sklearn.metrics import accuracy_score
    regr = linear_model.LinearRegression()  # initialize regressor
    regr.fit(X_train, y_train)  # fit training data
    diabetes_y_pred = regr.predict(X_test)  # make prediction on X test set
    print("Accurary of the test data using Linear regression is:" + str(accuracy_score(diabetes_y_pred, y_test)) + "%")


if __name__ == "__main__":
    main()
