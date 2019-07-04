import model
import torch
import numpy as np
from itertools import count
from settings import DEVICE,EPOCHS,GAME,EPISODE_LIFE,CLIP_REWARDS,SCALE,FRAME_STACK,STEPS
from atari_wrappers import wrap_deepmind,make_atari

import time
import matplotlib.pyplot as plt
from utils import log

env=make_atari(GAME)
env=wrap_deepmind(env,episode_life=EPISODE_LIFE,clip_rewards=CLIP_REWARDS,frame_stack=FRAME_STACK,scale=SCALE)

#
np.random.seed(1)
env.seed(0)


log_reward=log(abs_dir_path="F:/github/RL/Atari_DQN/log")
log_epic_step=log(abs_dir_path="F:/github/RL/Atari_DQN/log")
log_loss=log(abs_dir_path="F:/github/RL/Atari_DQN/log")

agent = model.Agent(DEVICE)

total_count=0
start_time=time.time()

done_=False

for i in count():
    if done_:
        break

    state = env.reset()

    epic_reward=0
    for t in count():
        total_count+=1

        if total_count%10000==0:
            end_time=time.time()
            print("1 w frame time: {}".format(end_time-start_time))
            start_time=end_time
            if total_count==STEPS:
                done_=True
                break

        action=agent.select_action(state)

        state_,reward,done,_=env.step(action.item())

        epic_reward+=reward

        reward=torch.tensor([reward],device=DEVICE)

        agent.remember(state,action,state_,reward)

        state=state_

        agent.learn()

        if done:
            log_reward.add(epic_reward)
            log_epic_step.add(t+1)

            print("episode: {} reward: {} step: {} memory: {}".format(i,epic_reward,t+1,len(agent.memory)))

            break

log_loss.add(agent.loss_hist)

fig=plt.figure()

ax1=fig.add_subplot(221)
ax2=fig.add_subplot(222)
ax3=fig.add_subplot(223)
ax1.plot(np.arange(0,len(log_loss.buf)),log_loss.buf)
ax2.plot(np.arange(0,len(log_reward.buf)),log_reward.buf)
ax3.plot(np.arange(0,len(log_epic_step.buf)),log_epic_step.buf)

plt.show()

log_epic_step.write(file_name="epic_step_3m.csv")
log_reward.write(file_name="reward_3m.csv")
log_loss.write(file_name="loss_3m.csv")
