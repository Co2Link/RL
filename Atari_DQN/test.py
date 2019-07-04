from utils import RingBuf
import random
import gym
from atari_wrappers import wrap_deepmind,make_atari
from settings import EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,N_ACTION,GAME,GAMMA,BATCH_SIZE,EPISODE_LIFE,CLIP_REWARDS,FRAME_STACK,SCALE,MEMORY_SIZE
import matplotlib.pyplot as plt
import numpy as np

import torch

env = make_atari(GAME)
env = wrap_deepmind(env, episode_life=EPISODE_LIFE, clip_rewards=CLIP_REWARDS, frame_stack=FRAME_STACK, scale=SCALE)
np.random.seed(1)
env.seed(0)

env.reset()
a=env.observation_space.sample()


l1=[1,2,3]

l2=2

l1.append(l2)

print(l1)