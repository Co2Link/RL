import model
import torch
import numpy as np
from itertools import count
from settings import DEVICE,EPOCHS,GAME,EPISODE_LIFE,CLIP_REWARDS,SCALE,FRAME_STACK
from atari_wrappers import wrap_deepmind,make_atari

import time
import matplotlib.pyplot as plt

env=make_atari(GAME)
env=wrap_deepmind(env,episode_life=EPISODE_LIFE,clip_rewards=CLIP_REWARDS,frame_stack=FRAME_STACK,scale=SCALE)

#
np.random.seed(1)
env.seed(0)


reward_hist=[]
epic_step_hist=[]

agent = model.Agent(DEVICE)

for i in range(EPOCHS):

    state = env.reset()

    epic_reward=0
    for t in count():

        action=agent.select_action(state)

        state_,reward,done,_=env.step(action.item())

        epic_reward+=reward

        reward=torch.tensor([reward],device=DEVICE)

        agent.remember(state,action,state_,reward)

        state=state_

        agent.learn()

        if done:
            reward_hist.append(epic_reward)
            epic_step_hist.append(t+1)

            print("episode: {} reward: {} step: {} memory: {}".format(i,epic_reward,t+1,len(agent.memory)))
            break
print('len of reward_hist: {}'.format(len(reward_hist)))
print('len of epic_step_hist: {}'.format(len(epic_step_hist)))

fig=plt.figure()

ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
print(agent.loss_hist)
ax1.plot(np.arange(0,len(agent.loss_hist)),agent.loss_hist)
ax2.plot(np.arange(0,len(reward_hist)),reward_hist)
ax3.plot(np.arange(0,len(epic_step_hist)),epic_step_hist)

plt.show()
