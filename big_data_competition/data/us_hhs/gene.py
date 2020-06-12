import pandas as pd 
import time
import numpy as np
import random
s= []
for i in range(451):
    for j in range(2900):
        num = random.uniform(1,10)
        s.append(num)
s = np.array(s).reshape(451,2900)
s = pd.DataFrame(s)
s.to_csv("d2.txt",header=None,index=None)
print("finished!!")
