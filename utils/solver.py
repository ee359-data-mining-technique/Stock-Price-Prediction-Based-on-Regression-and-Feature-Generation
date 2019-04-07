#-*- coding:utf-8 -*-

from sklearn.neighbors import KNeighborsClassifier


class KNNSolver:

    def __init__(self, distance, neighbors):
        self.distance = distance
        self.neighbors = neighbors

    def train_and_decode(self, train_X, test_X, train_y, test_y):
        print('Begin to train\ndistance type:\t{}\tnumber of neighbors:\t{}'.format(self.distance, self.neighbors))
        knn = KNeighborsClassifier(n_neighbors=self.neighbors, metric=self.distance)
        knn.fit(train_X, train_y)
        acc = knn.score(test_X, test_y)
        return acc




