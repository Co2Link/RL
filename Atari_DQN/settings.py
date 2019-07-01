import torch

DEVICE = torch.device('cuda')
SCREEN_WIDTH = 600
TARGET_UPDATE = 10
EPOCHS = 500
BATCH_SIZE = 128
GAMMA = 0.999
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
SCALE=True
MEMORY_SIZE=1000000