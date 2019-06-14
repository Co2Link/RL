import torch
import numpy as np
a=torch.randn(4,4)
b=torch.max(a,1)[1]
print(a)
a=a.cpu()
print(a)
print(b)
print(b[0])
