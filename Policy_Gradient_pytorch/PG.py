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

        # print(np.shape(pyt_obs))
        # print(np.shape(pyt_as))
        # time.sleep(10)

        # calculate the loss 这里已经验证过一遍，没问题
        acts_prob=self.net(pyt_obs).gather(1,pyt_as) # (batch,1)
        neg_log_prob=-torch.log(acts_prob)
        loss=torch.mean(neg_log_prob*pyt_rs)

        self.loss_hist.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.ep_rs,self.ep_as,self.ep_obs=[],[],[]
        return discounted_ep_rs_norm


    def _discount_and_norm_rewards(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        discounted_ep_rs -= np.mean(discounted_ep_rs)  # z-score标准化
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs

def train():
    DISPLAY_REWARD_THRESHOLD=-500

    model=PolicyGradient()
    ep_rs_hist=[]
    is_render=False
    for i_episode in range(100000):
        s = env.reset()
        while True:
            if is_render:env.render()

            a=model.choose_action(s)
            s_,r,done,info=env.step(a)
            model.store_transition(s,a,r)

            # if count%1000==0:
            #     # print(count)

            if done:

                ep_rs_sum=sum(model.ep_rs)
                ep_rs_hist.append(ep_rs_sum)

                if 'running_reward' not in globals():
                    global running_reward
                    running_reward=ep_rs_sum
                else:
                    running_reward=running_reward*0.99+ep_rs_sum*0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: is_render=True

                print('episode: {},reward: {}'.format(i_episode,ep_rs_sum))

                vt=model.learn()
                # if i_episode %20==0:
                #     plt.plot(vt)  # plot the episode vt
                #     plt.xlabel('episode steps')
                #     plt.ylabel('normalized state-action value')
                #     plt.show()
                break
            s = s_
    return [model.loss_hist,ep_rs_hist]




if __name__ == '__main__':
    # PG=PolicyGradient()
    # PG.ep_rs=[1.,2.,3.,45.,56.]
    # a=PG._discount_and_norm_rewards()
    # print(a)
    # print(np.shape(a))
    # state_sample=torch.randn(2,4) # (1,4)
    # PG.choose_action(state_sample)
    train()

