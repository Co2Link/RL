from utils import RingBuf
import random
import gym
from atari_wrappers import wrap_deepmind,make_atari
from settings import EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,N_ACTION,GAME,GAMMA,BATCH_SIZE,EPISODE_LIFE,CLIP_REWARDS,FRAME_STACK,SCALE,MEMORY_SIZE
import matplotlib.pyplot as plt
import numpy as np

import torch

# 若正常则静默

a = torch.tensor(1.)
# 若正常则静默
print(a.cuda())
# 若正常则返回 tensor(1., device='cuda:0')

from torch.backends import cudnn

# 若正常则静默
print(cudnn.is_available())
# 若正常则返回 True

print(cudnn.is_acceptable(a.cuda()))