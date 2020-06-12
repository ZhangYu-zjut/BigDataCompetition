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
    global temp
    temp = torch.zeros((day_length, p, m),dtype=torch.double)
    result = []
    global X
    for i in range(day_length):
        if (i==0):
            for inputs in loader.get_batches(data,batch_size=1,shuffle=False): 
                X = inputs[0]
                temp[0, :p, :] = X[0, :p, :]
                #print("temp0 is",temp)
        else:
            outputs = model(X)
            if(i==1):
                #print("X Y shape is ",X.shape,outputs.shape)
                #print("------predict x is---------",X)
                #print("------predict y is-------",outputs)
                for para in model.parameters():
                    #print(p)
                    pass
                #print("temp is::",temp)
            
            
            
            temp[i,0:p-1,:] = temp[i-1,1:p,:].clone()
            temp[i, p-1, :] = outputs[0,:]
            X[0,:p,:] = temp[i,:,:]
        X = X.cuda()
        #print("i is: {:d} | x shape: {:s}".format(i,X.shape))
        outputs = model(X)
        Prediction[i,0,:] = outputs[0,:]*loader.scale
        result.append(outputs)
    print("predict finished!")
    print("Prediction shape is",Prediction.shape)
    save("./data/output/new_%s_epoch_%d_window_%d_start_index_%d.csv"%(args.city_name,args.epochs,args.window,args.start_index),Prediction)

    return Prediction

def save(save_path,data):
    m = data.shape[0]
    result =[]
    for i in range(m):
        temp = data[i,0,:].detach().numpy()
        temp[temp<0] = 0
        temp = np.floor(temp)
        result.append(temp)
    res = pd.DataFrame(result)
    if os.path.exists(save_path)==False:
        #os.mkdir(save_path)
        pass
    res.to_csv(save_path,header=None,index=None)
    print("save successfully!")





