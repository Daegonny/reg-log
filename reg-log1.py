# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 5000
ALPHA = 0.1
M = 100

os.chdir("/home/daegonny/code/python/ia/reg-log/")
data = pd.read_csv("ex2data1.txt")
x = data[["x1","x2"]]
x=(x-x.min())/(x.max()-x.min())
x1 = x["x1"].values.tolist()
x2 = x["x2"].values.tolist()
y = data["y"].values.tolist()

theta_1 = 0.0
theta_2 = 0.0
theta_0 = 0.0

def sig(z):
    return 1/(1+np.exp(-z))

def eq(x):
    return (-theta_1*x - theta_0)/theta_2

def h(x1,x2):
    return sig(theta_1*x1 + theta_2*x2 + theta_0)


def j(h, y):
    if y == 1:
        return (-1)*np.log(h)    
    else:    
        return  (-1)*np.log(1-h) 
    

def step(value):
    if value >= 0.5:
        return 1
    else:
        return 0

costs = []

for epoch in range(EPOCHS):
    cost = 0
    del_theta_1 = 0
    del_theta_2 = 0
    del_theta_0 = 0    
    
    for row in zip(x1,x2,y):
        cost += j(h(row[0], row[1]), row[2])/M
        
        del_theta_1 += (h(row[0], row[1]) - row[2])* row[0]
        del_theta_2 += (h(row[0], row[1]) - row[2])* row[1]
        del_theta_0 += (h(row[0], row[1]) - row[2])
   
    theta_1 = theta_1 -(ALPHA/M) * del_theta_1
    theta_2 = theta_2 -(ALPHA/M) * del_theta_2    
    theta_0 = theta_0 -(ALPHA/M) * del_theta_0       
    costs.append(cost)
    
color = ['red' if item == 1 else 'blue' for item in y]
ret_x = [i/1000 for i in range(1000)]
ret_y = [eq(i) for i in ret_x]    
plt.scatter(x1, x2, s=50, c=color)    
plt.plot(ret_x, ret_y)
plt.show()    
#hits = 0    
    
#for row in zip(x1,x2,y):
#    if step(h(row[0],row[1])) == row[2]:
#        hits +=1

#print("Hits: "+str(hits))        
#print("Accuracy: "+str(hits/M))