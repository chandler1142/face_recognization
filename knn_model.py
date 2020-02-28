import csv
import os

from config.paths import csv_dir_indiv
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier

from utils import flattern


def knn_classifier(input, knn_path):
    knn = joblib.load(knn_path)
    prob = knn.predict_proba(input)
    pred = knn.predict(input)
    # print(max(distance[0][0]),pred)
    return pred, prob


def __generate_dataset(csv_dir_indiv):
    x_train = []
    y_train = []

    for root, dirs, files in os.walk(csv_dir_indiv):
        for f in files:
            rf = open(os.path.join(root, f), 'r')
            reader = list(csv.reader(rf))
            counter = 0
            for k in reader:
                if counter > 0:
                    read_path = k[3]
                    data = flattern(read_path)
                    x_train.append(data)
                    y_train.append(f.title().split(".")[0])
                else:
                    pass
                counter += 1
    return x_train, y_train


def knn_train():
    X_train, Y_train = __generate_dataset(csv_dir_indiv)
    knn = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    knn.fit(X_train, Y_train)
    joblib.dump(knn, './models/knn.model')
