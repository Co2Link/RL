import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

from utils import RingBuf
from settings import EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,N_ACTION,GAME,GAMMA,BATCH_SIZE,EPISODE_LIFE,CLIP_REWARDS,FRAME_STACK,SCALE,MEMORY_SIZE
from atari_wrappers import make_atari,wrap_deepmind

import cv2
import numpy as np


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4) # expect input shape (4,84,84)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)  # outpu shape (7x7x64)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512,N_ACTION)
        self.steps_done=0

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.linear1(x.view(x.size(0),-1))) # flatten, shape (1,3136)
        out = self.linear2(x)
        return out

class Agent():
    def __init__(self,device):
        self.device=device
        self.policy_net=DQN().to(self.device)
        self.target_net=DQN().to(self.device)
        self.optimizer=optim.RMSprop(self.policy_net.parameters())
        self.steps_done=0
        self.memory=RingBuf(size=MEMORY_SIZE)

    def select_action(self,state):
        sample=random.random()
        epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * self.steps_done/EPSILON_DECAY_STEPS
        self.steps_done += 1

        if sample <epsilon_threshold:
            return torch.tensor([[random.randrange(3)]],device=self.device,dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
    def remember(self,s,a,s_,r):
        self.memory.append([s,a,s_,r])
    def learn(self):
        if len(self.memory)<MEMORY_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        state_batch=np.concatenate([np.expand_dims(i[0],axis=0) for i in transitions])

        print(state_batch.shape)
        print(type(state_batch))



        # eval_q=self.policy_net(state_batch).gather(1,action_batch)

def DEBUG():
    # auto fire after reset,skip_frame=4,stack_frame=4,max_frame operation,scale operation,clipreward operation,episode_life,
    env=make_atari(GAME)
    env=wrap_deepmind(env,episode_life=EPISODE_LIFE,clip_rewards=CLIP_REWARDS,frame_stack=FRAME_STACK,scale=SCALE)


    env.reset()

    for i in range(100):
        img,reward,done,_=env.step(0) # img shape (84,84,4)
        img=np.array(img).transpose((2,0,1))[0]
        cv2.imshow('1',img)
        cv2.waitKey(0)
        if(done):
            break








if __name__ == '__main__':
    DEBUG()