#-*- coding:utf-8 -*-
import metric_learn as ml

class MLearner:
    def __init__(self, mltype):
        self.mltype = mltype

    def mmc(self, train_X, train_y, test_X):
        learner = ml.MMC_Supervised()
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def lmnn(self, train_X, train_y, test_X, k):
        learner = ml.LMNN(k=k)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def nca(self, train_X, train_y, test_X, dims):
        learner = ml.NCA(num_dims=dims)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def lfda(self, train_X, train_y, test_X, k, dims):
        learner = ml.LFDA(num_dims=dims, k=k)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def mlkr(self, train_X, train_y, test_X, dims):
        learner = ml.MLKR(num_dims=dims)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def itml(self, train_X, train_y, test_X, gamma):
        learner = ml.ITML(gamma=gamma)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def lsml(self, train_X, train_y, test_X):
        learner = ml.LSML()
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def sdml(self, train_X, train_y, test_X, bal, spa):
        learner = ml.SDML(balance_param=bal, sparsity_param=spa)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def rca(self, train_X, train_y, test_X, dims):
        learner = ml.RCA(num_dims=dims)
        train_X = learner.fit_transform(train_X, train_y)
        test_X = learner.transform(test_X)
        return train_X, test_X

    def learn(self, train_X, train_y, test_X, opt):
        if self.mltype == 'MMC':
            train_X, test_X = self.mmc(train_X, train_y, test_X)
        elif self.mltype == 'LMNN':
            k = opt.k
            train_X, test_X = self.lmnn(train_X, train_y, test_X, k)
        elif self.mltype == 'NCA':
            dims = opt.dims
            train_X, test_X = self.nca(train_X, train_y, test_X, dims)
        elif self.mltype == 'LFDA':
            k = opt.k
            dims = opt.dims
            train_X, test_X = self.lfda(train_X, train_y, test_X, k, dims)
        elif self.mltype == 'MLKR':
            dims = opt.dims
            train_X, test_X = self.mlkr(train_X, train_y, test_X, dims)
        elif self.mltype == 'ITML':
            gamma = opt.gamma
            train_X, test_X = self.itml(train_X, train_y, test_X, gamma)
        elif self.mltype == 'LSML':
            train_X, test_X = self.lsml(train_X, train_y, test_X)
        elif self.mltype == 'SDML':
            bal = opt.bal
            spa = opt.spa
            train_X, test_X = self.sdml(train_X, train_y, test_X, bal, spa)
        elif self.mltype == 'RCA':
            dims = opt.dims
            train_X, test_X = self.rca(train_X, train_y, test_X, dims)
        elif self.mltype == 'None':
            print('Not use any metric learn')
        else:
            raise ValueError
        return train_X, test_X