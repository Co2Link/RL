import gym
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

import settings
import cv2
import time

# GAME = "Breakout-v0"
# GAME = "BreakoutDeterministic-v4"
GAME = "CartPole-v0"

# Resize frames we grab from gym and convert to tensor
resizer = T.Compose([
    T.ToPILImage(),
    T.Resize(40, interpolation=Image.CUBIC),  # resize时插值的方法
    T.ToTensor()
])


# Start cartpole application through gym
def init():
    return gym.make(GAME).unwrapped


# Get cart location with respect to center
def get_cart_location(world, screen_width):
    world_width = world.x_threshold * 2
    scale = screen_width / world_width
    return int(world.state[0] * scale + screen_width / 2.0)


# Get screen tensor from gym application
def get_screen(world, screen_width, device):
    screen = world.render(mode='rgb_array').transpose((2, 0, 1))

    # cut the screen
    screen = screen[:, 160:320]

    view_width = 320
    cart_location = get_cart_location(world, screen_width)

    # cut the screen
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

    screen = screen[:, :, slice_range]

    # return a contiguous array in memory with the same value
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)  # shape (3,160,320)

    return resizer(screen).unsqueeze(0).to(device)  # shape(1,3,40,80)


def img_preprocesse(img):
    # convert to gray scale
    img = np.mean(img, axis=2).astype(np.uint8)
    # down-sampling
    img = img[::2, ::2]
    return img


def tranform_reward(reward):
    return np.sign(reward)


if __name__ == '__main__':
    world = init()
    world.reset()
    world.step(1)

    for i in range(100):
        start = time.time()
        screen, reward, done, _ = world.step(2)
        screen = img_preprocesse(screen)
        cv2.imshow('1', screen)
        cv2.waitKey(0)

    # screen = world.render('rgb_array')
    #
    # resizer = T.Compose([
    # 	T.ToPILImage
    # ])
    #
    # print(resizer(screen))
    #
    # print(screen.shape)
    #
    # cv2.imshow('1',screen)
    # cv2.waitKey(0)
    #

    # print(	get_screen(world,settings.SCREEN_WIDTH,device=settings.DEVICE).shape)
    world.close()
