import numpy as np
import math  
import random     
#functions list
'''
1.matrix_multiplication_1DX1D
2.matrix_multiplication_1DX2D
3.matrix_multiplication_2DX1D
4.matrix_multiplication_2DX2D
5.matrix_formation1D_row
6.matrix_formation1D
7.matrix_formation2D
8.Identity_matrix
9.Transpose
10.euclidean_norm
11.partial_pivot
12.artial_pivot2DX2D
13.gaussjordan
14.LU_decomposition
15.Forward substitution
16.Backward substitution
17.jacobi
18.jacobi2
19.Cholesky
20.Gauss seidel
21.conjugate Gradient
22.Inverse calculator
23.Jacobi for eigen systems
24.chisquare fit
25.chisquare linear regrssion
26.Power method
27.Inverse
28.Linear reg
29.Polinonial fit
30.LC Generator
31.Monte Carlo
'''


def matrix_multiplication_1DX1D(a,b):
    f=len(a)
    l=[0]
    for i in range(f):
        l[0]+=(a[i]*b[i])
    return l
def matrix_multiplication_1DX2D(a,b):
    f=len(a)
    d=len(b[0])
    l=[]
    for i in range(d):
        l.append(0)
    for i in range(d):
        for j in range(f):
            l[i]=l[i]+((a[j])*(b[j][i]))
    return l
def matrix_multiplication_2DX1D(a,b):
    f=len(a)
    k=len(a[0])
    l=[]
    row=[0]
    for i in range(f):
        l.append(row)
    for i in range(f):
        for j in range(k):
            l[i][0]+=(a[i][j]*b[j])
    return l
def matrix_multiplication_2DX2D(a,b):
    f=len(a)
    d=len(b[0])
    h=len(a[0])    
    l=[]
    for i in range(f):
        row=[]
        for j in range(d):
            row.append(0.0)
        l.append(row)
    for i in range(f):
        for j in range(d):
            for k in range(h):
                l[i][j]=l[i][j]+((a[i][k])*(b[k][j]))
    return l
def matrix_formation1D_row(a):
    a1=a.read()
    a2=a1.split(' ')
    n=len(a2)
    l=[0.0 for i in range(n)]
    for i in range(n):
        l[i]=float(a2[i])
    return l
def matrix_formation1D(a):
    l = []
    for line in a:
        l1=line
        l.append(l1)
    n=len(l)
    l2=[0.0 for j in range(n)]
    for i in range(n):
        l2[i]=float(l[i])
    return l2
#it is for reading text file and forming a matrix from it
def matrix_formation2D(a):
    l = []
    for line in a:
        l1 = line.split()
        l.append(l1)
    m=len(l)
    n=len(l[0])
    l2=[[0.0 for j in range(n)]for i in range(m)]
    for i in range(m):
        for j in range(n):
            l2[i][j]=float(l[i][j])
    return l2
def Identity_matrix(n):
    #n is the dimension
    b=[]
    for i in range(n):
        row=[]
        for j in range(n):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        b.append(row)
    return b

def transpose(a):
    m=len(a)
    n=len(a[0])
    l=[[0.0 for j in range(m)] for i in range(n)]
    for i in range(m):
        for j in range(n):
            l[j][i]=a[i][j]
    return l
def euclidean_norm(x):
    n = 0
    for i in range (len(x)):
        n+=((x[i])**2)
    norm=(n**(0.5))
    return norm
def partial_Pivot(n,b,r):
    #by
    i=r
    k=0
    l=0
    h=len(n)
    if n[r][r]<1e-10:      
        for i in range(r,(h-1)):
            if n[r][r]==0 and abs(n[i+1][r])>abs(n[r][r]):
                for j in range(3):
                    k=n[i+1][j]
                    n[i+1][j]=n[r][j]
                    n[r][j]=k
                    l=b[i+1]
                    b[i+1]=b[r]
                    b[r]=l
            else:
                continue
    return n,b

def partial_pivot(n,b,r):
    #by
    h=len(n)
    if n[r][r]<1e-10:      
        for i in range(r+1,h):
            if n[r][r]==0 and abs(n[i][r])>abs(n[r][r]):
                n[i],n[r]=n[r],n[i]
                b[i],b[r]=b[r],b[i]
    return n,b
def partial_pivot2DX2D(n,b,r):
    i=r
    k=0
    l=0
    h=len(n)
    if n[r][r]< 1e-10:      
        for i in range(r,(h-1)):
            if n[r][r]<1e-10 and (abs(n[i+1][r])>abs(n[r][r])):
                for j in range(h):
                    k=n[i+1][j]
                    n[i+1][j]=n[r][j]
                    n[r][j]=k
                    l=b[i+1][j]
                    b[i+1][j]=b[r][j]
                    b[r][j]=l
            else:
                continue
    return n,b
def gaussjordan(A,B):
    a=np.array(A)
    b=np.array(B)
    n=len(b)
    for k in range(n):
        partial_pivot(a,b,k)
        # Make pivot row diagonal element to 1 by division
        pivot = a[k,k]
        for j in range(k,n):
            a[k,j] /= pivot
        b[k] /= pivot
        # Elimination loop
        for i in range(n):
            if i == k or abs(a[i,k]) < 1e-10 :
                continue
            else:
                factor = a[i,k]
                for j in range(k,n):
                    a[i,j] -= (factor * a[k,j])
                b[i] -= (factor * b[k])
    return b, a

def LU_decomposition(m,b):
    n=len(m)
    #decomposing the given matrix into Upper and lower triangular matrix
    for j in range(n):
        partial_pivot(m,b,j)
        for i in range(n):
            if i<=j:
                
                sum=0
                for k in range(i):
                    sum+=((m[i][k])*(m[k][j]))
                m[i][j]=(m[i][j]-sum)
            else:
                sum=0
                for k in range(j):
                    sum+=((m[i][k])*(m[k][j]))
                m[i][j]=((m[i][j]-sum)/m[j][j])
    return m,b
def forward_substitution(m,b):
    global y
    #solving Ly=b where y=Ux for the equation ax=b
    #here m is L
    n=len(m)
    y=[0.0 for i in range(n)]
    for i in range(n): 
        sum=0.0
        for j in range(n):
            sum+=((m[i][j])*(y[j]))
        y[i]+=(b[i]-sum)
    return y


def backward_substitution(m,y):
    #solving Ux=y for the equation ax=b
    #here m is U
    n=len(m)
    x=[0.0 for k in range(n)] 
    for i in reversed(range(n)):
        sum=0.0
        for j in reversed(range(n)):
            if j>i:
                sum+=((m[i][j])*(x[j]))
        x[i]+=((y[i]-sum)/(m[i][i]))
    return x
#if np.allclose(x, X, rtol=1e-5):
            #break
def inf_norm(x1,x2):
    sum=0.0
    n=len(x1)
    for i in range(n):
        diff=abs(x1[i]-x2[i])
        sum+=(diff)
    return sum
def jacobi(a,b,x,tolerance,iterations):
    #here x is the initial guess of x
    #here k is the iteration limit
    n=len(b)
    q=tolerance
    Error=[]
    itr=[]
    #x = np.zeros_like(b)
    for z in range(iterations):
        X=[0.0 for i in range(n)]
        for i in range(n):
            s=0.0
            for j in range(n):
                if i!=j:
                    s+=((a[i][j])*x[j])
            X[i]=(1/(a[i][i]))*(b[i]-s)
            
            if X[i] == X[i-1]:
                break
        
        #checking the covergance condition
        xT=np.transpose(X)
        ax=np.dot(a,xT)
        if np.allclose(x,X,atol=1e-04, rtol=0.):
            break
        error = np.dot(a, x) - b
        sum=0.0
        for i in range(len(x)):
            sum+=((error[i])**2)
        err=np.sqrt(sum)
        Error.append(err)
        itr.append(z)
        x=X
    return X,Error,itr
def jacobi2(a,b,x,eps):
    #here x is the initial guess of x
    #here k is the iteration limit
    n=len(a)
    m=len(a[0])
    X=np.array([0.0 for i in range(n)])
    #X = np.zeros_like(b)
    k=0
    while inf_norm(x,X)>=eps:
        for i in range(n):
            s=0.0
            for j in range(m):
                if i!=j:
                    s+=((a[i][j])*x[j])
            X[i]=(1/(a[i][i]))*(b[i]-s)
            
            #if X[i] == X[i-1]:
                #break
        x=X
        k+=1
        #checking the covergance condition
        '''
        xT=np.transpose(X)
        ax=np.dot(a,xT)
        if np.allclose(ax,b,atol=1e-10):
            break
        '''
    return X


def cholesky(A):
    #for returning L
    n = len(A)
    L = [[0 for i in range(n)]
         for j in range(n)]
    #for computing l_ii
    for i in range(n):
        s=0.0
        for j in range(i):
            s+= L[i][j]**2
        L[i][i] = math.sqrt(A[i][i] - s)
    #for computing l_ij
        for k in range(i+1,n):
            s = 0.0
            for j in range(i):
                s+=L[i][j]*L[k][j]
            L[k][i]=(1/L[i][i])*(A[k][i]-s)
    return L



def gauss_seidel(a,b,x,tolerance,iterations):
    n = len(a)
    a=np.array(a)
    x=np.array(x)
    error=[]
    itr=[]
    for k in range(iterations):
        #creating another vector for the updating purpose of x
        X = x.copy()
        for i in range(len(a[0])):
            s1=0.0
            s2=0.0
            x[i] = (b[i] - np.dot(a[i, :i], x[:i]) - np.dot(a[i, (i+1):], X[(i+1):])) / a[i, i]
        err= np.linalg.norm(x - X, ord=np.inf) / np.linalg.norm(x, ord=np.inf)   
        error.append(err)
        itr.append(k)
        if err< tolerance:
            break
            '''
            for j in range(i+1,len(a[0])):
                s1 += (a[i][j]*X[j])
            for j in range(1,i):
                s2 += (a[i][j]*x[j])
            X[i] = (b[i] - s1 - s2) / a[i][ i]
            
            if X[i] == x[i-1]:
                break

        #if np.allclose(x, X, rtol=1e-5):
            #break
        
    #checking the covergance condition
        #xT=np.transpose(X)
        #ax=np.dot(a,xT)
        if np.allclose(x,X,atol=1e-4, rtol=0.):
            break
        x = X
        '''
    #calculating error
    #error = np.dot(A, x) - B
    return x,error,itr

def gauss_seidel2(a,b,x,tolerance,iterations):
    n = len(a)
    #x=np.zeros_like(b)
    error=[]
    itr=[]
    for k in range(iterations):
        #creating another vector for the updating purpose of x
        X=np.zeros_like(b)
        s1=0.0
        s2=0.0
        for i in range(len(a[0])):
            for j in range(i+1,len(a[0])):
                s1 += (a[i][j]*X[j])
            for j in range(1,i):
                s2 += (a[i][j]*x[j])
            X[i] = (b[i] - s1 - s2) / a[i][ i]
            
            if X[i] == x[i-1]:
                break

        
    #checking the covergance condition
        #xT=np.transpose(X)
        #ax=np.dot(a,xT)
        if np.allclose(x,X,atol=tolerance, rtol=0.):
            break
        x = X
    
    #calculating error
    #error = np.dot(A, x) - B
    return x

def gauss_seidel3(a,b,x,eps):
    n = len(a)
    X = np.zeros_like(x)
    k=0
    while inf_norm(x,X) >= eps:
        if k!=0:
            x=X
            
        for z in range(n):
        #creating another vector for the updating purpose of x
            for i in range(len(a[0])):
                s1=0.0
                s2=0.0
                for j in range(i+1,len(a[0])):
                    s1 += (a[i][j]*X[j])
                for j in range(1,i):
                    s2 += (a[i][j]*x[j])
                X[i] = (b[i] - s1 - s2) / a[i][ i]
        #if np.allclose(x, X, rtol=1e-5):
            #break
    #checking the covergance condition
        #xT=np.transpose(X)
        #ax=np.dot(a,xT)
        #if np.allclose(x,X,atol=1e-10):
            #break
        k+=1
    #calculating error
    #error = np.dot(A, x) - B
    return x
def conjugate_gradient(a,b,x,tolerance,iterations):
    a=np.array(a)
    b=np.array(b)
    x=np.array(x)
    # r is the direction of steepest descent
    r=b-(a.dot(x))
    R=r.copy()
    #here R is the new residual for updating purpose
    error=[]
    itr=[]
    for i in range(iterations):
        aR=a.dot(R)
        if np.dot(R, aR)==0:
            break
        alpha=np.dot(R, r)/np.dot(R, aR)
        #alpha is similar to learning rate
        x+=(alpha*R)
        r=b-(a.dot(x))
        error.append(np.sqrt(np.sum((r**2))))
        itr.append(i)
        if np.sqrt(np.sum((r**2))) < tolerance:
            break
        else:
            beta= -np.dot(r,aR)/np.dot(R,aR)
            R = r + (beta* R)
    #returning solution x after the process
    return x,error,itr


def inverse_calculator(a, method,x,tolerance,iterations):
    '''
    x = initial guess of x
    '''
    n = len(a)
    inv = np.zeros((len(a), len(a)))
    #res_list_comb = []
    for i in range(n):
        b = [0.0 for i in range(n)]
        b[i] = 1.0
        X,err,itr = method(a, b, x,tolerance,iterations)
        for j in range(1, len(b)):
            inv[:, i] = X
        #invT.append(X)
        #invT=np.array(invT, dtype=np.double)
        #res_list_comb.extend(res_list)
    #inv = np.transpose(invT)
    return inv,err,itr
def inverse_calculator2(A, method,x,eps):
    n = len(A)
    inv = []
    #res_list_comb = []
    for i in range(n):
        b = [0.0 for i in range(n)]
        b[i] = 1
        X,err,itr = method(A, b, x,eps)
        inv.append(X)
        #res_list_comb.extend(res_list)
    inv = np.transpose(np.array(inv))
    return inv,err,itr
def Jacobi_for_eigen_system(A):
    #for largest off-diagonal element
    n=len(A)
    def maxind(a):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j])>=Amax:
                    Amax = abs(A[i,j])
                    k = i
                    l = j
        return Amax, k,l

    # to make A[k,l] = 0 by rotation and define rotation matrix
    def transformation(a,p,k,l):
        a_diff=a[l,l]-a[k,k]
        if abs(a[k,l])< abs(a_diff)*1e-30:
            tan=a[k,l]/a_diff
        else:
            phi=a_diff/(2*a[k,l])
            tan=1/(abs(phi)+np.sqrt(phi**2+1))
            if phi<0:
                tan=-tan
        cos=1/np.sqrt(tan**2+1)
        sin=tan*cos
        tau=sin/(1+cos)
        temp=a[k,l]
        a[k,l]=0
        a[k,k]=a[k,k]-tan*temp
        a[l,l]=a[l,l]+tan*temp
        for i in range(k):
            temp=A[i,k]
            a[i,k]=temp-sin*(a[i,l]+tau*temp )
            a[i,l]+=sin*(temp-tau*a[i,l])
        for i in range(k+1,l):
            temp=A[k,i]
            a[k,i]=temp-sin*(a[i,l]+tau*a[k,i])
            a[i,l]+=sin*(temp-tau*a[i,l])
        for i in range(l+1, n):
            temp=A[k,i]
            a[k,i]=temp-sin*(a[l,i]+tau*temp)
            a[l,i]+=sin*(temp-tau*a[l,i])
        for i in range(n):
            temp=p[i,k]
            p[i,k]=temp-sin*(p[i,l]-tau*p[i,k])
            p[i,l]+=sin*(temp-tau*p[i,l])
    p=np.identity(n)
    for i in range(5*(n**2)):
        Amax,k,l=maxind(A)
        if Amax < 1e-9:
            return np.diagonal(A)
        transformation(A,p,k,l)

def chisquare(ob_bin,ex_bin,constrain = 1):
    #chisqaure for observed and expecteed results
    degrees_of_freedom = len(ex_bin)-constrain
    chisqr = 0
    for j in range(len(ex_bin)):
        if ex_bin[j]<= 0:
            print("The Error observed is in expected number of instances of bin")
        temp = ob_bin[j]-ex_bin[j]
        chisqr=chisqr+ temp*temp/ex_bin[j]
    return chisqr

def chisquare_for_2_datasets(bin1,bin2,constrain=1):
    #here bin1 and bin2 are two datasets
    degrees_of_freedom=len(bin1)-constrain
    chisqr = 0
    for j in range(len(bin1)):
        if bin1[j] == 0 and bin2[j] == 0:
            degrees_of_freedom -= 1
        else:
            chisqr=chisqr+pow((bin1[j]-bin2[j]),2)/(bin1[j]+bin2[j])
def chi_square_Linear_reg(x, y, sigma):
    #here sigma is the error in y
    a=0.0
    b=0.0
    chisquare=0.0
    n=len(y)
    m=len(x)
    S=0.0
    Sx=0.0
    Sy=0.0
    Sxx=0.0
    Sxy=0.0
    global covariance_ab
    global erroraglobal
    global error_b
    y_p=[0 for x in range(n)]
    for i in range(m):
        y_p[i]=a * b*x[i]
        chisquare=chisquare+((y[i]-y_p[i])/sigma[i])**2
        S+=1/sigma[i]**2
        Sx+=x[i]/sigma[i]**2
        Sy+=y[i]/sigma[i]**2
        Sxx+=(x[i]/sigma[i])**2
        Sxy+=x[i]*y[i]/sigma[i]**2
    delta=S*Sxx-(Sx)**2
    a=(Sxx*Sy-Sx*Sxy)/delta
    b=(S*Sxy-Sx*Sy)/delta
    covariance_ab=-Sx/delta
    error_a=np.sqrt(Sxx/delta)
    error_b=np.sqrt(S/delta)
    return a,b,covariance_ab,error_a,error_b
def power_method(A,x,y,n):
    #A is the given matrix
    #A should be a numpy matrix/array
    #x is the initial guess of eigen vector
    #x=c1v1+c2v2+c3v3+........
    #Assume a simple possible dominating eigenvector with norm equal to 1
    #y be any vector not orthogonal to v1
    #v1 is the dominating eigen vector
    #n=number of iterations
    import numpy as np
    x1=x
    x2=x
    #x1=np.array([0.0 for i in range(len(A))])
    #x2=np.array([0.0 for i in range(len(A))])
    #u1  and loopis the initial guess of eigen vector
    for i in range(n):
        x1=np.matmul(A, x1)
        norm=np.linalg.norm(x1)
        #normalizing u1
        X1=x1/norm
    for i in range(n-1):
        x2=np.matmul(A, x2)
        norm=np.linalg.norm(x2)
        #normalizing u1
        X2=x2/norm
    #calculating the approximate dominating eigenvalue
    eigvalue=np.dot(x1,y)/np.dot(x2,y)
    #
    #calculation of approximate dominating eigenvector
    eigvect=X2

    
    
    
    return eigvalue, eigvect
def power_method_including_non_dominant(A,x,y,n,m):
    #m is the number of non-dominant eigen vectors needed
    eigval=[]
    eigvec=[]
    for i in range(m):
        eigvalue,eigvect = power_method(A, x, y, n)
        A=A-(eigvalue*(np.matmul(eigvect, (np.transpose(eigvect)))))
        eigval.append(eigvalue)
        eigvec.append(eigvect)
    return eigval,eigvec
def jackknife(A):
    #you should convert the given dataset into a matrix
    #and then evaluate it's number of columns to know in 
    #how many rows we need to apply the processes of mean and all
    a=np.transpose(A)
    m=len(a)
    n=len(a[0])
    #n is the number of elements in each column
    #m is the number of columns(number of 1D arrays inside the 2D array)
    #bcz I am taking row here as columns
    y_mean_k=[]
    y_mean_square_k=[]
    for i in range(m):
        y_k_each=[]
        y_k_each_square=[]
        for k in range(n):
            y_sum=0.0
            for j in range(n):
                if j!=k:
                   y_sum+=a[i][j]
            
            y_k_each.append(y_sum/(n-1))
            y_k_each_square.append(((y_sum/(n-1))*(y_sum/(n-1))))
        y_mean_k.append(y_k_each)
        y_mean_square_k.append(y_k_each_square)
    y_mean_jk=[]
    y_mean_square_k_mean=[]
    for i in range(m):
        y_sum=0.0
        y_sum_sqaure=0.0
        for j in range(n):
            y_sum+=y_mean_k[i][j]
            y_sum_sqaure+=y_mean_square_k[i][j]
        y_mean_jk.append(y_sum/n)
        y_mean_square_k_mean.append(y_sum_sqaure/n)
        #here each element of y_mean_jk represent mean of 
        #each column(where each element of the column is y_k bar)
    sigma_jk_square=np.subtract(y_mean_square_k_mean, np.matmul(y_mean_jk,y_mean_jk))
    sigma_square=(n-1)*(sigma_jk_square)
    #here sigma and sigma_jk_square are the variances
    return y_mean_jk,sigma_jk_square,sigma_square     

def inverse(n):
    h=len(n)
    b=[]
    for i in range(h):
        row=[]
        for j in range(h):
            if i==j:
                row.append(1)
            else:
                row.append(0)
        b.append(row)
    for r in range(h):
        partial_pivot(n, b, r)
        pivot=n[r][r]
        for c in range(h):
            n[r][c]=n[r][c]/pivot
            b[r][c]=b[r][c]/pivot
        for r2 in range(h):
            if r2==r or n[r2][r]==0:
                continue
            else:
                factor=n[r2][r]
                for c2 in range(h):
                    n[r2][c2]=n[r2][c2]-(n[r][c2]*factor)
                    b[r2][c2]=b[r2][c2]-(b[r][c2]*factor)
    return b
def Linear_reg(x, y, sigma):
    a=0.0
    b=0.0
    chisquare=0.0
    n=len(x)
    m=len(x)
    S=0.0
    Sx=0.0
    Sy=0.0
    Sxx=0.0
    Sxy=0.0
    global covariance_ab
    global erroraglobal
    global error_b
    y_p=[0 for x in range(n)]
    for i in range(n):
        y_p[i]=a * b*x[i]
        chisquare=chisquare+((y[i]-y_p[i])/sigma[i])**2
        S+=1/sigma[i]**2
        Sx+=x[i]/sigma[i]**2
        Sy+=y[i]/sigma[i]**2
        Sxx+=(x[i]/sigma[i])**2
        Sxy+=x[i]*y[i]/sigma[i]**2
        delta=S*Sxx-(Sx)**2
        a=(Sxx*Sy-Sx*Sxy)/delta
        b=(S*Sxy-Sx*Sy)/delta
        covariance_ab=-Sx/delta
        error_a=np.sqrt(Sxx/delta)
        error_b=np.sqrt(S/delta)
    return a,b,covariance_ab,error_a,error_b
from scipy import linalg

def polynomial_fit(x,y,delta_y,order):
    x=np.array(x)
    y=np.array(y)
    delta_y=np.array(delta_y)
    k=order+1
    #k is the number of parameters
    X=[]
    for i in range(k):
        X.append([])
        for j in range(k):
            X[i].append(np.sum(((x**(i+j)))/(delta_y**2)))
    Y=[]
    for i in range(k):
        Y.append([])
        Y[i].append(np.sum(np.multiply((x**i),y)/(delta_y**2)))
    B=[]
    for i in range(k):
        B.append([])
        for j in range(k):
            if i==j:
                B[i].append(1)
            else:
                B[i].append(0)
    inv =linalg.inv(np.array(X))
    solution=np.dot(inv,Y)
    return solution

def LC_Generator(seed,a,c,m,n):
    # seed represents the starting number
    # n represents the total number of random numbers to be generated
    # a,c,m represents the parameters of the equation given
    #m is used for mod fn
    randomgen = np.zeros(n)
    randomgen[0] = seed
    itr=[]
    itr.append(0)
    for i in range(1, n):
        randomgen[i] = ((a*randomgen[i-1]) + c) % m
        itr.append(i)
    return randomgen/m,itr
def Monte_Carlo(f, n):
    # f reprsents the input function
    # n represents the number of points for integration
    x,itr= LC_Generator(3,687,2,186845,n)
    x=np.array(x)
    sum = 0.0
    for i in range(n):
        sum += f(x[i])
    sum/=n
    return sum

        
'''
import matplotlib.pyplot as plt
x=[0]
fig = plt.figure()
ax = plt.axes()
a,b,cov_ab,err_a,err_b = Linear_reg(x,y,sigma)
x = np.linspace(0, 10, 1000)
plt.plot(x, a + b*x)
plt.show()
y
plt.plot(x)
plt.scatter(y_predicted,y_test)
plt.plot(xp,yp,linewidth=1, color='r', label="Bandgaps are equal")
plt.grid()
plt.legend()
plt.ylabel("Predicted Bandgaps")
plt.xlabel("Experimental Bandgaps")
plt.title("Prediction and experimental bandgaps for test data")
plt.show()
    '''
