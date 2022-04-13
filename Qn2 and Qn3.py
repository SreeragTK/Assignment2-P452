import numpy as np
import mylibrary as ml
#answering the second question
#generatin random numbers for the first condition
r1,itr1=ml.LC_Generator(1000,65,0.0,1021,30)
print("Random numbers ganerated corrsponding to the first conditon(a=65, m=1021 is(N choosen as 30):", r1)
#generatin random numbers for the second condition
r2,itr2=ml.LC_Generator(3,65,10,16381,30)
print("Random numbers ganerated corrsponding to the second conditon(a=572, m=16381 is(N choosen as 30):", r2)
#Calculation of pi using points thorowing method
#Defining a function for it
def pi_calculation(n):
    pts1,itr1 = ml.LC_Generator(3,3007,654946,18,n)
    pts2,itr12= ml.LC_Generator(5,5003,956755,21,n)
    inside_number=0
    for i in range (n):
        p=[]
        p.append(pts1[i])
        p.append(pts2[i])
        k=pts1[i]-0.5
        l=pts2[i]-0.5
        norm=((k*k)+(l*l))**(0.5)
        if norm<0.5: 
            inside_number+=1
    pi=4*((inside_number)/(len(pts1)))
    return pi
pi=pi_calculation(1000000)
print("The value of pi using throwing points method is :",pi)
def f1(x):
    return np.sqrt(1-(x**2))
#Finding the value of poi using Monte carlo method
Pi=4*(ml.Monte_Carlo(f1,10000))
print("The value of poi using Monte carlo method is :",Pi)


#Answering qn 3
#Defining function for finding the volume of the given Steinmentz solid
#volume is the integral of (4*(1-x^2)) in the region -1 to 1
def f2(x):
    k=4*(1-x**2)
    return k
v=ml.Monte_Carlo(f2,10000)
#Here the integration has to be done from -1 to 1
#But the calculated "v" is the integral corresponding to 0 to 1 region
# So the actual volume is 2 times v
volume=2*v
print("Volume of Steinmets solid calculated using Monte Carlo merthod is:", volume)
'''
Results are
Random numbers ganerated corrsponding to the first conditon(a=65, m=1021 is(N choosen as 30): [0.97943193 0.66307542 0.09990206 0.49363369 0.08619001 0.60235064
 0.15279138 0.93143976 0.54358472 0.33300686 0.64544564 0.9539667
 0.00783546 0.5093046  0.10479922 0.81194907 0.77668952 0.48481881
 0.51322233 0.35945152 0.36434868 0.68266405 0.37316357 0.25563173
 0.61606268 0.04407444 0.86483839 0.21449559 0.94221352 0.24387855]
Random numbers ganerated corrsponding to the second conditon(a=572, m=16381 is(N choosen as 30): [1.83139003e-04 1.25144985e-02 8.14052866e-01 9.14046761e-01
 4.13649960e-01 8.87857884e-01 7.11372932e-01 2.39851047e-01
 5.90928515e-01 4.10963922e-01 7.13265368e-01 3.62859410e-01
 5.86472132e-01 1.21299066e-01 8.85049753e-01 5.28844393e-01
 3.75496001e-01 4.07850559e-01 5.10896771e-01 2.08900556e-01
 5.79146572e-01 6.45137659e-01 9.34558330e-01 7.46901899e-01
 5.49233869e-01 7.00811916e-01 5.53385019e-01 9.70636713e-01
 9.19968256e-02 9.80404127e-01]
The value of pi using throwing points method is : 3.333336
The value of poi using Monte carlo method is : 3.1349318194322504
Volume of Steinmets solid calculated using Monte Carlo merthod is: 5.317031043210735
'''