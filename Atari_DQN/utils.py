import numpy as np
import gym
import settings
import cv2

def get_env():
    return gym.make(settings.GAME)

# def img_preprocesse(img,resize=None):
#     # convert to gray scale
#     img = np.mean(img, axis=2).astype(np.uint8)
#     # down-sampling
#     # img = img[::2, ::2]
#     if resize:
#         img=cv2.resize(img,(84,84))
#     return img

def img_preprocesse(img):
    # convert to gray scale
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # resize
    img=cv2.resize(img,(84,84),interpolation=cv2.INTER_AREA)
    print("before: ",img.shape)
    return img

class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
    def sample(self,batch_size):
        pass

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