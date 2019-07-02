import numpy as np
import gym
import settings
import random
import cv2

def get_env():
    return gym.make(settings.GAME)

# actual buf size is size+1
class RingBuf:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        self.len=size+1

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    # code style need to be improved
    def sample(self,batch_size):
        assert (self.end+1)%self.len==self.start # sample when buf is full
        if self.data[self.len-1]==None: # the last item is None when it just full
            return random.sample(self.data[self.start:self.end],batch_size)
        else:
            return random.sample(self.data,batch_size) # sample a from list that len=size+1
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]