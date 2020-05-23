
from test import Pre_Data_utility
# X shape(1,p,m)
# Y shape(1,m)
# T shape(day,p,m)
# y 可能需要保存一下
def inference(loader,data,model,args):
    p = args.window
    m = data[0].size[2]
    print("m is",m)
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


