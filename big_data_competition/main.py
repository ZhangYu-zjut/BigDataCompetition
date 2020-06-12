#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import argparse
import math
import time

import torch
import torch.nn as nn
from models import AR, VAR, GAR, RNN
from models import CNNRNN, CNNRNN_Res
import numpy as np
import sys
import os
from utils import *
import Optim


class rmsle_loss(nn.Module):
    def __init__(self):
        super(rmsle_loss,self).__init__()
        self.L1 = nn.L1Loss(reduction='mean')
        self.L2 = nn.MSELoss(reduction='mean')

    def forward(self,output,Y,scale):
        #a=torch.tensor([[1.0,2,3,4,5,6,7,8,9]],dtype=torch.float).to(device)
        loss = torch.sqrt( (torch.log1p(output * scale) - torch.log1p(Y * scale)).pow(2).mean() )
        return loss

def evaluate(loader, data, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;

    for inputs in loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        output = model(X);
        from sklearn.preprocessing import MinMaxScaler
        #mm = MinMaxScaler()
        #output = mm.inverse_transform(output)
        #print("output is:",output)

        if predict is None:
            predict = output.cpu();
            test = Y.cpu();
        else:
            predict = torch.cat((predict,output.cpu()));
            test = torch.cat((test, Y.cpu()));

        scale = loader.scale.expand(output.size(0), loader.m)
        total_loss += evaluateL2(output * scale , Y * scale ).item()
        total_loss_l1 += evaluateL1(output * scale , Y * scale ).item()
        n_samples += (output.size(0) * loader.m);

    #print("saple,rse",n_samples,loader.rse)
    rse = math.sqrt(total_loss / n_samples)/loader.rse
    rae = (total_loss_l1/n_samples)/loader.rae
    correlation = 0;

    predict = predict.data.numpy();
    Ytest = test.data.numpy();
    sigma_p = (predict).std(axis = 0);
    sigma_g = (Ytest).std(axis = 0);
    mean_p = predict.mean(axis = 0)
    mean_g = Ytest.mean(axis = 0)
    index = (sigma_g!=0);

    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis = 0)/(sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    # root-mean-square error, absolute error, correlation
    return rse, rae, correlation;


def train(loader, data, model, criterion, optim, batch_size, args):
    model.train();
    total_loss = 0;
    n_samples = 0;
    counter = 0
    for inputs in loader.get_batches(data, batch_size, False):
        counter += 1
        X, Y = inputs[0], inputs[1]
        show = 0
        if(show ==1):
            print("normal y is:",Y)
            scale = loader.scale
            print("original y is:",Y*scale)
            print("scale is",scale)
            print("X,Y shape is: ",X.size(),Y.size())
        model.zero_grad();
        output = model(X);
        if(args.output_print):
            print("input X is",X)
            print("output is:",output)
        
        scale = loader.scale.expand(output.size(0), loader.m)
    
        # original l2 loss
        loss = criterion(output * scale, Y * scale);
        
        # rmsle loss--add the following 2 line code
        #criteon = rmsle_loss()
        #loss = criteon(output,Y,scale)

        loss.backward();
        #print("loss is::: ",loss)
    
        #print("X i s:",X)
        #print("output is: ",output)
    
        optim.step();
        total_loss += loss.item();
        n_samples += (output.size(0) * loader.m);
        #print("n_sample are: ",n_samples)
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='Epidemiology Forecasting')
# --- Data option
parser.add_argument('--data', type=str, required=True,help='location of the data file')
parser.add_argument('--train', type=float, default=0.6,help='how much data used for training')
parser.add_argument('--valid', type=float, default=0.2,help='how much data used for validation')
parser.add_argument('--model', type=str, default='AR',help='model to select')
parser.add_argument('--log_pre',type=str,default=True,help='use log preprocess or not')
# --- CNNRNN option
parser.add_argument('--sim_mat', type=str,help='file of similarity measurement (Required for CNNRNN, CNN)')
parser.add_argument('--hidRNN', type=int, default=50, help='number of RNN hidden units')
parser.add_argument('--residual_window', type=int, default=4,help='The window size of the residual component')
parser.add_argument('--ratio', type=float, default=1.,help='The ratio between CNNRNN and residual')
parser.add_argument('--output_fun', type=str, default=None, help='the output function of neural net')
# --- Logging option
parser.add_argument('--save_dir', type=str,  default='./save',help='dir path to save the final model')
parser.add_argument('--save_name', type=str,  default='tmp', help='filename to save the final model')
# --- Optimization option
parser.add_argument('--optim', type=str, default='adam', help='optimization method')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--epochs', type=int, default=100,help='upper epoch limit')
parser.add_argument('--clip', type=float, default=1.,help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay (L2 regularization)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',help='batch size')
# --- Misc prediction option
parser.add_argument('--horizon', type=int, default=12, help='predict horizon')
parser.add_argument('--window', type=int, default=24 * 7,help='window size')
parser.add_argument('--metric', type=int, default=1, help='whether (1) or not (0) normalize rse and rae with global variance/deviation ')
parser.add_argument('--normalize', type=int, default=0, help='the normalized method used, detail in the utils.py')

parser.add_argument('--seed', type=int, default=54321,help='random seed')
parser.add_argument('--gpu', type=int, default=None, help='GPU number to use')
parser.add_argument('--cuda', type=str, default=True, help='use gpu or not')
parser.add_argument('--city_name',type=str,default='city_A',help='choose the city')
parser.add_argument('--start_index',type=int,default=0,help='set the start index of window')
parser.add_argument('--output_print',type=int,default=0,help='whether print the model output')
parser.add_argument('--result_vis',type=int,default=0,help='whether display the prediction trend')

args = parser.parse_args()
print(args);
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if args.model in ['CNNRNN', 'CNN'] and args.sim_mat is None:
    print('CNNRNN/CNN requires "sim_mat" option')
    sys.exit(0)

args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

Data = Data_utility(args);

model = eval(args.model).Model(args, Data);

if args.cuda:
    model.cuda()

nParams = sum([p.nelement() for p in model.parameters()])
print('* number of parameters: %d' % nParams)

criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();


best_val = 1000000000;
optim = Optim.Optim(
    model.parameters(), args.optim, args.lr, args.clip, weight_decay = args.weight_decay,
)

# At any point you can hit Ctrl + C to break out of training early.
try:
    print('begin training');
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(Data, Data.train, model, criterion, optim, args.batch_size, args)
        val_loss, val_rae, val_corr = evaluate(Data, Data.valid, model, evaluateL2, evaluateL1, args.batch_size);
        print('| end of epoch {:3d} | time: {:.6f}s | train_loss {:5.8f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))
        # Save the model if the validation loss is the best we've seen so far.
        if val_loss < best_val:
            best_val = val_loss
            model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
            with open(model_path, 'wb') as f:
                torch.save(model.state_dict(), f)
            print('best validation');
            test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
            print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
model_path = '%s/%s.pt' % (args.save_dir, args.save_name)
with open(model_path, 'rb') as f:
    model.load_state_dict(torch.load(f));
test_acc, test_rae, test_corr  = evaluate(Data, Data.test, model, evaluateL2, evaluateL1, args.batch_size);
print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))


sys.path.append(r'./data/')
from data.predict import inference
from data.pre_utils import Pre_Data_utility

#from data import Pre_Data_utility

pre_data = Pre_Data_utility(args)

#for parameters in model.parameters():
    #print(parameters)

#print("model is",model)
prediction = inference(pre_data, pre_data.train, model, args)

print("prediction result is:\n",prediction)
