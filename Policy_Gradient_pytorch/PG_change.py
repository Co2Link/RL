import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt

import time

env = gym.make('MountainCar-v0')
# env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_STATES = env.observation_space.shape[0]
N_ACTIONS = env.action_space.n

np.random.seed(1)
torch.manual_seed(1)


print('N_STATES: ',N_STATES)
print('N_ACTIONS: ',N_ACTIONS)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(N_STATES,10)
        self.fc1.weight.data.normal_(0, 0.3)   # initialization
        self.fc2=nn.Linear(10,N_ACTIONS)
        self.fc2.weight.data.normal_(0, 0.3)   # initialization

    def forward(self, x):
        x=self.fc1(x)
        x=torch.tanh(x)
        x=self.fc2(x)
        act_prob=F.softmax(x)
        return act_prob

class PolicyGradient(object):
    def __init__(
            self,
            learning_rate=0.01,     # 调整lr后顺利在 episode500之后 reward -500
            reward_decay=0.995,
    ):
        self.net=Net()
        self.lr=learning_rate
        self.gamma=reward_decay
        self.optimizer=torch.optim.Adam(self.net.parameters(),lr=self.lr)

        self.ep_obs,self.ep_as,self.ep_rs=[],[],[]

        self.max_path_length=20000
        self.min_steps_per_patch=5000
        self.ep_steps_list=[]

        self.loss_hist=[]


    def choose_action(self,x):
        x=torch.unsqueeze(torch.FloatTensor(x),0)
        act_prob=self.net.forward(x)
        action=np.random.choice(range(act_prob.size()[1]),p=act_prob.data.numpy().ravel())
        return action
    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    def learn(self):
        discounted_ep_rs_norm=self._discount_and_norm_rewards()
        pyt_obs=torch.FloatTensor(self.ep_obs)
        pyt_as=torch.unsqueeze(torch.LongTensor(self.ep_as),1)
        pyt_rs=torch.unsqueeze(torch.FloatTensor(discounted_ep_rs_norm),1)

        # calculate the loss
        acts_prob=self.net(pyt_obs).gather(1,pyt_as) # (batch,1)
        neg_log_prob=-torch.log(acts_prob)
        loss=torch.mean(neg_log_prob*pyt_rs)

        self.loss_hist.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_rs,self.ep_as,self.ep_obs=[],[],[]
        self.ep_steps_list=[]
        return discounted_ep_rs_norm


    def _discount_and_norm_rewards(self):
        index=0
        discounted_rs=[]
        for ep_steps in self.ep_steps_list:
            ep_rs=self.ep_rs[index:index+ep_steps]

            discounted_ep_rs = np.zeros_like(ep_rs)
            running_add = 0
            for t in reversed(range(0, len(ep_rs))):
                running_add = running_add * self.gamma + ep_rs[t]
                discounted_ep_rs[t] = running_add
            discounted_ep_rs -= np.mean(discounted_ep_rs)  # normalize , z-score标准化
            discounted_ep_rs /= np.std(discounted_ep_rs)
            index+=ep_steps
            discounted_rs.extend(discounted_ep_rs) # now

        return discounted_rs

    def _discount_and_norm_rewards_simple(self): # 这个比上面那个耗时多几千倍，平均耗时几秒
        max_step = len(self.ep_rs)
        dis_rs = [np.sum(np.power(self.gamma, np.arange(max_step - t)) * self.ep_rs[t:]) for t in range(max_step)] # compute the discounted reward with gamma
        mean = np.mean(dis_rs, axis=0)
        std = np.std(dis_rs, axis=0)
        adv = (dis_rs - mean) / std
        return adv

def train():
    DISPLAY_REWARD_THRESHOLD=-100 # 在reward高于这个值时渲染动画
    model=PolicyGradient()
    ep_rs_hist=[]
    is_render=False
    for i_episode in range(100000):
        s = env.reset()
        ep_steps=0
        while True:
            if is_render:env.render()

            a=model.choose_action(s)
            s_,r,done,info=env.step(a)
            model.store_transition(s,a,r)

            ep_steps += 1
            # 在收到done信号或steps数超过设定值时结束该回合
            if done or ep_steps >= model.max_path_length:

                # log and display
                model.ep_steps_list.append(ep_steps)
                ep_rs_sum=sum(model.ep_rs[-ep_steps:])
                ep_rs_hist.append(ep_rs_sum)

                if 'running_reward' not in globals():
                    global running_reward
                    running_reward=ep_rs_sum
                else:
                    running_reward=running_reward*0.99+ep_rs_sum*0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: is_render=True

                print('episode: {},ep_reward: {},running_reward: {}'.format(i_episode,ep_rs_sum,running_reward))

                # 存储足够多的transition后进行一次学习
                if sum(model.ep_steps_list) > model.min_steps_per_patch:
                    vt=model.learn()

                break
            s = s_
    return [model.loss_hist,ep_rs_hist]



if __name__ == '__main__':

    train()

