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



def tranform_reward(reward):
    return np.sign(reward)