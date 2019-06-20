import numpy as np

gamma=0.995

def _discount_and_norm_rewards(ep_rs):
    discounted_ep_rs = np.zeros_like(ep_rs)
    running_add = 0
    for t in reversed(range(0, len(ep_rs))):
        running_add = running_add * gamma + ep_rs[t]
        discounted_ep_rs[t] = running_add
    discounted_ep_rs -= np.mean(discounted_ep_rs)  # z-score标准化
    discounted_ep_rs /= np.std(discounted_ep_rs)
    return discounted_ep_rs

def _discount_and_norm_rewards_simple(ep_rs):
    max_step=len(ep_rs)
    dis_rs = [np.sum(np.power(gamma, np.arange(max_step - t)) * ep_rs[t:]) for t in range(max_step)]
    mean=np.mean(dis_rs,axis=0)
    std=np.std(dis_rs,axis=0)
    adv=(dis_rs-mean)/std
    return adv

a=np.random.rand(10)
print(_discount_and_norm_rewards(a))
print(_discount_and_norm_rewards_simple(a))
print(a)
