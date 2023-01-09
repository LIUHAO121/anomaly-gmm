import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def jiecheng(n):
    if n==1:
        return n
    else:
        return n * jiecheng(n-1)

def exp_zk(x,n=54):
    output = 1
    for i in range(1,n+1):
        output += x**i / jiecheng(i)
        print(output)
    return output


def mtdf1(x):
    output = -math.log(exp_zk(-x))
    return output

def mtdf2(x):
    output = math.log(1+x)
    return output


s1 = []
s2 = []
x = []
for i in range(1,1000):
    s1.append(mtdf1(i*0.1))
    s2.append(mtdf2(i*0.1))
    x.append(i*0.1)
s1 = pd.Series(s1)
s2 = pd.Series(s2)

fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(211)
ax.plot(x,s1, color="red",label="-log(1-x+x**2/2 ... )")
ax.legend()
ax = fig.add_subplot(212)
ax.plot(x,s2,color="blue",label="log(1+x)")
ax.legend()
plt.savefig("mtdf.png")