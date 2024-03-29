import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000

env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

device=torch.device('cuda:0')
DISPLAY_REWARD_THRESHOLD=-2000

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self,is_ddqn=True,is_cuda=False):
        if is_cuda:
            self.eval_net, self.target_net = Net().cuda(), Net().cuda()
        else:
            self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.is_ddqn=is_ddqn
        self.loss_hist=[]

        self.is_cuda=is_cuda
        if is_ddqn:
            self.name='ddqn'
        else:
            self.name='dqn'

    def choose_action(self, x):
        if self.is_cuda:
            x = torch.unsqueeze(torch.FloatTensor(x), 0).cuda()
        else:
            x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition =np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter updatea
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        if self.is_cuda:
            b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda()
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int)).cuda()
            b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2]).cuda()
            b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).cuda()
        else:
            b_s = torch.FloatTensor(b_memory[:, :N_STATES])
            b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
            b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
            b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])


        # q_eval w.r.t the action in experience
        print('gather: ',np.shape(b_s))
        print('gather: ',np.shape(b_a))
        time.sleep(10)

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        if self.is_ddqn:
            actions_value=self.eval_net.forward(b_s_)
            action=torch.max(actions_value,1)[1].view(BATCH_SIZE,1) # action was chosen by eval_net
            q_next=self.target_net(b_s_).detach().gather(1,action) # action was evaluated by target_net
        else:
            q_next = self.target_net(b_s_).detach().max(1)[0].view(BATCH_SIZE,1)  # detach from graph, don't backpropagate

        q_target = b_r + GAMMA * q_next   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)

        self.loss_hist.append(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def plot_loss(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arrange(len(self.loss_hist)),self.loss_hist)
        plt.ylabel('loss')
        plt.xlabel('training steps')
        plt.show()


def train(model):
    print('training -{}-'.format(model.name))
    ep_r_hist=[]
    is_render=False
    for i_episode in range(300):
        s = env.reset()
        ep_r = 0
        while True:
            if is_render:
                env.render()
            a = model.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            model.store_transition(s, a, r, s_)

            ep_r += r
            if model.memory_counter > MEMORY_CAPACITY:
                model.learn()
            if done:
                ep_r_hist.append(ep_r)
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
                # if 'running_reward' not in globals():
                #     global running_reward
                #     running_reward=ep_r
                # else:
                #     running_reward=running_reward*0.99+ep_r*0.01
                # if running_reward>DISPLAY_REWARD_THRESHOLD:is_render=True
                break
            s = s_
    return [model.loss_hist,ep_r_hist]


## 开启cuda之后速度更慢：待调查

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ddqn=DQN()
    # dqn=DQN(is_ddqn=False)

    loss_ddqn,r_ddqn=train(ddqn)
    # loss_dqn,r_dqn=train(dqn)

    plt.figure(1)

    ax1=plt.subplot(1,2,1)
    ax2=plt.subplot(1,2,2)

    plt.sca(ax1)
    plt.plot(np.array(loss_ddqn),c='r',label='ddqn')
    # plt.plot(np.array(loss_dqn),c='b',label='dqn')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.xlabel('training steps')
    plt.grid()

    plt.sca(ax2)
    plt.plot(np.array(r_ddqn),c='r',label='ddqn')
    # plt.plot(np.array(r_dqn),c='b',label='dqn')
    plt.legend(loc='best')
    plt.ylabel('reward')
    plt.xlabel('training steps')
    plt.grid()

    plt.show()

