import torch

DEVICE = torch.device('cuda')
TARGET_REPLACE_ITER=1000
EPOCHS = 100000
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1
EPSILON_END = 0.1
EPSILON_DECAY_STEPS = 10**6
N_ACTION = 4

# GAME = "Breakout-v4"
# GAME = "BreakoutDeterministic-v4" # image shape (210,160,3)
# GAME = "CartPole-v0"
GAME="BreakoutNoFrameskip-v4"

EPISODE_LIFE=True
CLIP_REWARDS=True
FRAME_STACK=True
SCALE=False # need more than 100G RAM to store 100M frame under float32
MEMORY_SIZE=10000
LR=0.00025