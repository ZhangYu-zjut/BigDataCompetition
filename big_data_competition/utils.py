import torch
import numpy as np
from torch.autograd import Variable
from math import *

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon
        self.args = args
        fin = open(args.data)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        n,m = self.rawdat.shape
        epsilon = 1
        #print("original data:",self.rawdat)
        for i in range(n):
            for j in range(m):
                if(args.log_pre==False):
                    break
                if(self.rawdat[i][j]==0):
                    self.rawdat[i][j] += epsilon
                    #print("log 10!!!!!!!!!!!!!!!!!!!!")
                #self.rawdat[i][j] = log(self.rawdat[i][j],10)
        #print("new data",self.rawdat)
        #print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = args.normalize
        self.scale = np.ones(self.m)
        self._normalized(self.normalize)

        self._split(int(args.train * self.n), int((args.train+args.valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()

        #compute denominator of the RSE and RAE
        self.compute_metric(args)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(args.sim_mat, delimiter=','))
        #print("original adj matrix is",self.adj)
        # normalize
        #print("raw sim is",self.adj)
        den = self.adj.sum(dim=0)
        den[den==0] = 1
        rowsum = 1. / torch.sqrt(den)
        #rowsum = 1. /torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        #print("new adj is",self.adj)
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()

    def compute_metric(self, args):
        #use the normal rmse and mae when args.metric == 0
        if (args.metric == 0):
            self.rse = 1.
            self.rae = 1.
            return

        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))



    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        #normalized by the maximum value of entire matrix.
        if (normalize == 1):
            adjust_factor = 1
            self.scale = adjust_factor *self.scale * (np.mean(np.abs(self.rawdat))) * 2
            self.dat = adjust_factor * self.rawdat / (np.mean(np.abs(self.rawdat)) * 2)
            print("matrix max is:",np.abs(self.rawdat))

        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                #print("row max is",np.abs(self.rawdat[:,i]))
                self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max(np.abs(self.rawdat[:,i]))
        # add a new normalization: log10(x)
        if(normalize ==3):
            for i in range(self.m):
                self.sacle[i] = self.dat[:,i]/log(self.rawdat[:,i],10)
                self.dat[:,i] = log(self.rawdat[:,i],10) 


    def _split(self, train, valid, test):
        global offset
        offset = 15
        if(self.args.city_name=="city_E"):
            offset = 23
        train_set = range(self.P+self.h-1+self.args.start_index, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

        if (train==valid):
            self.valid = self.test


    def _batchify(self, idx_set, horizon):

        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i,:self.P,:] = torch.from_numpy(self.dat[start:end, :])
            Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])

        return [X, Y]

    def get_batches(self, data, batch_size, shuffle=True):
        inputs = data[0]
        targets = data[1]
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()
            model_inputs = Variable(X)

            data = [model_inputs, Variable(Y)]
            yield data
            start_idx += batch_size
