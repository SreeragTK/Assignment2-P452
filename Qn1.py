import numpy as np
import mylibrary as ml
import matplotlib.pyplot as plt
f1=open("assign2fit.txt","r")
a1=ml.matrix_formation2D(f1)
a1=ml.transpose(a1)
#print(a1)
x=a1[0]
y=a1[1]
print(x)
print(y)
Dy=[0.05 for i in range(len(y))]
para_list=ml.polynomial_fit(x,y,Dy,order=3)
print("The solution set corresponding to [a0,a1,a2,a3,a4] is:",para_list)
#The results are,
'''
The solution set corresponding to [a0,a1,a2,a3,a4] is: [[  0.57465867]
 [  4.72586144]
 [-11.12821778]
 [  7.66867762]]
'''