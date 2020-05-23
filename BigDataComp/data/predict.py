#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pre_utils
import torch
import os
import pandas as pd
import numpy as np
def inference(loader,data,model,args):
    p = args.window
    m = loader.m
    day_length = 30
    Prediction = torch.zeros((day_length, 1, m))
    temp = torch.zeros((day_length, p, m))
    result = []
    global X
    for i in range(day_length):
        if (i==0):
            for inputs in loader.get_batches(data,batch_size=1,shuffle=False): 
                X = inputs[0]
                temp[0, :p, :] = X[0, :p, :]
        else:
            outputs = model(X)
            temp[i,0:p-1,:] = temp[i,1:p,:]
            temp[i, p - 1, :] = outputs[0,:]
            X[0,:p,:] = temp[i,:,:]
        X = X.cuda()
        #print("i is: {:d} | x shape: {:s}".format(i,X.shape))
        outputs = model(X)
        Prediction[i,0,:] = outputs[0,:]*loader.scale
        result.append(outputs)
    print("predict finished!")
    print("Prediction shape is",Prediction.shape)
    save("./data/output/new_%s_window_%d.csv"%(args.city_name,args.window),Prediction)
    return Prediction

def save(save_path,data):
    m = data.shape[0]
    result =[]
    for i in range(m):
        temp = data[i,0,:]
        result.append(temp.detach().numpy())
    res = pd.DataFrame(result)
    if os.path.exists(save_path)==False:
        #os.mkdir(save_path)
        pass
    res.to_csv(save_path,header=None,index=None)
    print("save successfully!")





