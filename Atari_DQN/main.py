import model
import torch
import numpy as np
from itertools import count
from settings import DEVICE,SCREEN_WIDTH,TARGET_UPDATE,EPOCHS,GAME,EPISODE_LIFE,EPSILON_DECAY_STEPS,CLIP_REWARDS,SCALE,FRAME_STACK
from atari_wrappers import wrap_deepmind,make_atari

env=make_atari(GAME)
env=wrap_deepmind(env,episode_life=EPISODE_LIFE,clip_rewards=CLIP_REWARDS,frame_stack=FRAME_STACK,scale=SCALE)

durations=[]

agent = model.Agent(DEVICE)

for i in range(EPOCHS):
    img = env.reset()

    img=np.array(img).transpose((2,0,1))

    img = torch.from_numpy(img).unsqueeze(0).to(DEVICE)

    for t in count():
        action=agent.select_action(img)

        state,reward,done,state_=env.step(action)

        agent.remember(state,action,state_,reward)

        state=state_

        agent.learn()

        if done:
            durations.append(t+1)
            break
