from utils import RingBuf
import random
import gym
from atari_wrappers import wrap_deepmind,make_atari
from settings import EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,N_ACTION,GAME,GAMMA,BATCH_SIZE,EPISODE_LIFE,CLIP_REWARDS,FRAME_STACK,SCALE,MEMORY_SIZE


env_2=gym.make('Breakout-v0')

env=make_atari(GAME)
env=wrap_deepmind(env,episode_life=EPISODE_LIFE,clip_rewards=CLIP_REWARDS,frame_stack=FRAME_STACK,scale=SCALE)

a=env.action_space
print(a)
print(env_2.state_space())