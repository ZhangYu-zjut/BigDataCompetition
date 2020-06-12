# -*- coding: utf-8 -*-



import torch
import numpy as np
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Pre_Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, args):
        self.cuda = args.cuda
        self.P = args.window
        self.h = args.horizon

        fin = open(args.data)
        self.rawdat = np.loadtxt(fin,delimiter=',')
        print('data shape', self.rawdat.shape)
        if args.sim_mat:
            self.load_sim_mat(args)

        if (len(self.rawdat.shape)==1):
            self.rawdat = self.rawdat.reshape((self.rawdat.shape[0], 1))
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = args.normalize
        self.scale = np.ones(self.m)
        self._normalized(self.normalize)

        self._split(int(args.train * self.n))

        self.scale = torch.from_numpy(self.scale).float()

        #compute denominator of the RSE and RAE
        #self.compute_metric(args)

        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)

    def load_sim_mat(self, args):
        self.adj = torch.Tensor(np.loadtxt(args.sim_mat, delimiter=','))
        # normalize
        rowsum = 1. / torch.sqrt(self.adj.sum(dim=0))
        self.adj = rowsum[:, np.newaxis] * self.adj * rowsum[np.newaxis, :]
        self.adj = Variable(self.adj)
        if args.cuda:
            self.adj = self.adj.cuda()

    def _normalized(self, normalize):
        if (normalize == 0):
            self.dat = self.rawdat
        #normalized by the maximum value of entire matrix.
        if (normalize == 1):
            self.scale = self.scale * (np.mean(np.abs(self.rawdat))) * 2
            self.dat = self.rawdat / (np.mean(np.abs(self.rawdat)) * 2)

        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.m):
                row_max = np.max(np.abs(self.rawdat[:,i]))
                if(row_max ==0):
                    row_max += 0.001
                self.scale[i] = row_max  #np.max(np.abs(self.rawdat[:,i]))
                #print("i,max",i,np.max(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / row_max

    # 这个要修改
    def _split(self, train):
        self.train = self._batchify([self.n], self.h)

    def _batchify(self, idx_set, horizon):
        n = 1
        X = torch.zeros((n, self.P, self.m))
        # 主要这里该
        for i in range(n):
            end = idx_set[i] - self.h + +1
            start = end - self.P
            print("start is{:d} | end is {:d}".format(start,end))
            X[i,0:self.P,:] = torch.from_numpy(self.dat[start:end, :])
        return [X]

    def get_batches(self, data, batch_size=1, shuffle=True):
        inputs = data[0]
        X = inputs
        if (self.cuda):
            X = X.cuda()
            model_inputs = Variable(X)
            data = [model_inputs]
            yield data

# X shape(1,p,m)
# Y shape(1,m)
# T shape(day,p,m)
# y 可能需要保存一下
def inference(loader,data,model,args):
    p = args.window
    m = data[0].size[2]
    day_length = 30
    Prediction = torch.zeros((day_length, 1, m))
    temp = torch.zeros((day_length, p, m))
    result = []
    global X
    for i in range(day_length):
        if (i==0):
            for inputs in loader.get_batches(data, batch_size, True):
                X = inputs[0]
                temp[0, :p, :] = X[0, :p, :]
        else:
            outputs = model(X)
            temp[i,0:p-1,:] = temp[i,1:p,:]
            temp[i, p - 1, :] = outputs[0,:]
            X = temp[i,:,:]
        outputs = model(X)
        Prediction[i,0,:] = outputs[0,:]*loader.scale
        result.append(outputs)
    print("predict finished!")
    return Prediction
    """
        for inputs in loader.get_batches(data, batch_size, True):
            # X shape (1,p,m)
            if(i==0):
                X = inputs[0]
            else:
                temp[i,:p-1,:] = X[,,]
                temp[i, p, :] = outputs
            outputs = model(X)
            print("norm predict is",outputs)
            print("original predict is",outputs*loader.scale)
            Prediction[i,0,:] = outputs[0,:]
            X = X + Prediction[i,0,:]
    """
