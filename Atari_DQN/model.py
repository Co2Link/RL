import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

from utils import RingBuf
from settings import EPSILON_START,EPSILON_END,EPSILON_DECAY_STEPS,N_ACTION,GAME,GAMMA,BATCH_SIZE,EPISODE_LIFE,CLIP_REWARDS,FRAME_STACK,SCALE,MEMORY_SIZE,TARGET_REPLACE_ITER,LR
from atari_wrappers import make_atari,wrap_deepmind

import cv2
import numpy as np
import time

Transition = namedtuple('Transition',('state','action','state_','reward'))


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
        self.optimizer=optim.RMSprop(self.policy_net.parameters(),lr=LR)
        self.steps_done=0
        self.memory=RingBuf(size=MEMORY_SIZE)
        self.n_action=N_ACTION
        self.loss_func=nn.MSELoss()

        self.loss_hist=[]
        self.learn_step_counter=0
    def select_action(self,state):

        state=self._state2tensor(state)

        sample=random.random()
        epsilon_threshold = EPSILON_START-(EPSILON_START-EPSILON_END)*min(self.steps_done,EPSILON_DECAY_STEPS)/EPSILON_DECAY_STEPS
        self.steps_done += 1

        if sample <epsilon_threshold:
            return torch.tensor([[random.randrange(self.n_action)]],device=self.device,dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1,1)
    def remember(self,s,a,s_,r):
        self.memory.append(Transition(s,a,s_,r))
    def learn(self):
        if len(self.memory)<MEMORY_SIZE:
            return
        if self.learn_step_counter%TARGET_REPLACE_ITER==0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.learn_step_counter+=1

        transitions = self.memory.sample(BATCH_SIZE)

        batch=Transition(*zip(*transitions))

        # # DEBUG :check image
        # batch_state=np.array(batch.state_).transpose((0,3,1,2))
        # for i in batch_state:
        #     cv2.imshow('1',i[0])
        #     cv2.waitKey(0)

        batch_state=self._state2tensor(batch.state) # shape(128,4,84,84)
        batch_action=torch.cat(batch.action)        # shape(128,1)
        batch_reward=torch.cat(batch.reward)        # shape(128,)
        batch_state_=self._state2tensor(batch.state_) # shape(128,4,84,84)


        q_eval=self.policy_net(batch_state).gather(1,batch_action)

        action_value=self.policy_net(batch_state_)
        action=torch.max(action_value,1)[1].view(BATCH_SIZE,1)
        q_target=self.target_net(batch_state_).detach().gather(1,action)

        q_target=batch_reward+GAMMA*q_target


        loss=self.loss_func(q_eval,q_target)
        self.loss_hist.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    # turn lazy_frame into tensor
    def _state2tensor(self,state):
        if isinstance(state,tuple):
            state=np.array(state).transpose((0,3,1,2)) # (N,H,W,C) to (N,C,H,W)
            state=state.astype(np.float32)/255.0
            state=torch.from_numpy(state).to(self.device)
        else:
            state=np.array(state).transpose((2,0,1)) # (H,W,C) to (C,H,W)
            state=state.astype(np.float32)/255.0
            state=torch.from_numpy(state).unsqueeze(0).to(self.device)
        return state

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