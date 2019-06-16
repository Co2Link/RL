import torch
import numpy as np
a=torch.randn(4,4)
a=np.array([[1,2,3],
            [2,3,4],
            [3,4,5]])
b=np.array([[1,0,0],
            [0,1,0],
            [0,0,1]])
print(a*b)