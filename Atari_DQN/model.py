import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.conv1=nn.Conv2d()
