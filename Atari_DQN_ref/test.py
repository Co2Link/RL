from collections import deque,namedtuple
import random
import torch
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

mydeque = deque(maxlen=10)

for i in range(10):
    mydeque.append(Transition(state=torch.tensor([i]),action=torch.tensor([i+1]),next_state=torch.tensor([i+2]),reward=torch.tensor([i+3])))

ran=random.sample(mydeque,3)
k=Transition(*zip(*ran))

k_cat=torch.cat(k.state)

print(k.state)
print(k_cat)
print(type(k.state))
print(type(k_cat))

