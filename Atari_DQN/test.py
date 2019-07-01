from utils import RingBuf
import random

buf=RingBuf(size=10)

for i in range(11):
    buf.append(i)

print(list(buf))