import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

#https://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)
    
    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    #get convert
    covmat = np.cov(X.T)
    #get mean
    #append X and y by column
    z = np.append(X, y, axis=1)
    X_col_num = X.shape[1]
    #Make five Matrix Mi to store info of X labeled by y = i 
    M1 = np.empty((0, X_col_num), int)
    M2 = np.empty((0, X_col_num), int)
    M3 = np.empty((0, X_col_num), int)
    M4 = np.empty((0, X_col_num), int)
    M5 = np.empty((0, X_col_num), int)
    
    for row in z:
        #check y in each row, then discard y in row and append row in Mi
        if(row.item(X_col_num) == 1):
            
            row = np.asmatrix(row[:-1])
            M1 = np.append(M1, row, axis=0)
        elif(row.item(X_col_num) == 2):
            row = np.asmatrix(row[:-1])
            M2 = np.append(M2, row, axis=0)
        elif(row.item(X_col_num) == 3):
            row = np.asmatrix(row[:-1])
            M3 = np.append(M3, row, axis=0)
        elif(row.item(X_col_num) == 4):
            row = np.asmatrix(row[:-1])
            M4 = np.append(M4, row, axis=0)
        elif(row.item(X_col_num) == 5):
            row = np.asmatrix(row[:-1])
            M5 = np.append(M5, row, axis=0)
    #make means matrix
    means = np.empty((0, X_col_num), int)
    for i in range(1, 6):
        #generate variable Mi by i
        M = vars()['M' + str(i)]
        # compute Mi's mean by row
        M_mu = M.mean(axis=0)
        #append mean to means matrix
        means = np.append(means, M_mu, axis=0)
    # make k x d to d x k
    means = means.T
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    #append X and y by column to get z
    z = np.append(X, y, axis=1)
    X_col_num = X.shape[1]
    #Make five Matrix Mi to store info of X labeled by y = i 
    M1 = np.empty((0, X_col_num), int)
    M2 = np.empty((0, X_col_num), int)
    M3 = np.empty((0, X_col_num), int)
    M4 = np.empty((0, X_col_num), int)
    M5 = np.empty((0, X_col_num), int)
    
    #seperate X's row bases on yi and store them in Mi. i =[1, 5]
    for row in z:
        if(row.item(X_col_num) == 1):
            row = np.asmatrix(row[:-1])
            M1 = np.append(M1, row, axis=0)
        elif(row.item(X_col_num) == 2):
            row = np.asmatrix(row[:-1])
            M2 = np.append(M2, row, axis=0)
        elif(row.item(X_col_num) == 3):
            row = np.asmatrix(row[:-1])
            M3 = np.append(M3, row, axis=0)
        elif(row.item(X_col_num) == 4):
            row = np.asmatrix(row[:-1])
            M4 = np.append(M4, row, axis=0)
        elif(row.item(X_col_num) == 5):
            row = np.asmatrix(row[:-1])
            M5 = np.append(M5, row, axis=0)
    
    #get each means and covmats of each Mi
    means = np.empty((0, X_col_num), int)
    covmats = list()
    for i in range(1, 6):
        M = vars()['M' + str(i)]
        M_mu = M.mean(axis=0)
        M_cov = np.cov(M.T)
        covmats.append(M_cov)
        means = np.append(means, M_mu, axis=0)
        
    # k x d to d x k
    means = means.T

    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    assert(Xtest.shape[0] == ytest.shape[0]), 'row number must be same between Xtest and ytest'
    #init ys_mle
    ys_mle = np.empty((Xtest.shape[0],0), float)
    #get MLE for each yi, i = [1,5] && i is integer
    #and append it in ys_mle matrix
    for mean in means.T:  # get each mean in means
        y_mle = list()
        mean = mean.T     # dx1 
        
        for X_row in Xtest:
            X_row = np.asmatrix(X_row)
            X_row = X_row.T   # dx1
            yelem = pdf_multivariate_gauss(X_row, mean, covmat)
            y_mle.append(yelem)
        y_mle = np.asarray(y_mle)
        y_mle = np.asmatrix(y_mle)
        ys_mle = np.append(ys_mle, y_mle.T, axis=1)
   
    #generate ypred N x 1 matrix by check y_mle in ys_mle 
    ypred = list()
    for row in ys_mle:
        ypred.append(np.argmax(row)+1)
    ypred = np.asarray(ypred)
    ypred = np.asmatrix(ypred)
    ypred = ypred.T
    
    # compute acc by equation: acc = correct / total
    assert(ypred.shape[0] == ytest.shape[0]), 'row number must be same between ypred and ytest'
    correct = 0
    total = 0
    for i in range(ytest.shape[0]):
        total += 1
        if ypred.item((i, 0)) == ytest.item((i,0)):
            correct += 1
    acc = correct / total
    
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    assert(Xtest.shape[0] == ytest.shape[0]), 'row number must be same between Xtest and ytest'
    #init ys_mle
    ys_mle = np.empty((Xtest.shape[0],0), float)
    i = 0
    #get MLE for each yi, i = [1,5] && i is integer
    #and append it in ys_mle matrix
    for mean in means.T:
        y_mle = list()
        mean = mean.T     #dx1 
        for X_row in Xtest: # get each row in Xtest
            X_row = np.asmatrix(X_row)
            X_row = X_row.T   # dx1
            yelem = pdf_multivariate_gauss(X_row, mean, covmats[i])
            y_mle.append(yelem)
        y_mle = np.asarray(y_mle)
        y_mle = np.asmatrix(y_mle)
        ys_mle = np.append(ys_mle,y_mle.T, axis=1) # add y_mle.T to ys_mle by column
        i += 1
        
    #generate ypred N x 1 matrix by checkingy_mle in ys_mle     
    ypred = list()
    for row in ys_mle:
        ypred.append(np.argmax(row)+1)
    ypred = np.asarray(ypred)
    ypred = np.asmatrix(ypred)
    ypred = ypred.T # make 1 x N to N x 1
    
    # compute acc by equation: acc = correct number / total number
    assert(ypred.shape[0] == ytest.shape[0]), 'row number must be same between ypred and ytest'
    correct = 0
    total = 0
    for i in range(ytest.shape[0]):
        total += 1
        if ypred.item((i, 0)) == ytest.item((i,0)):
            correct += 1
    acc = correct / total
    
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # IMPLEMENT THIS METHOD
    X_t = np.transpose(X)   #X transpose
    X_tX = np.matmul(X_t,X) # Xt times X
    X_tX_inv = inv(X_tX)    # inverse X_tX
    w = X_tX_inv.dot(X_t).dot(y)                                             
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1
    # IMPLEMENT THIS METHOD  
    X_t = np.transpose(X)   #X transpose
    X_tX = X_t.dot(X)
    I = np.identity(X_tX.shape[0])
    lambd_I = lambd * I
    part_1 = inv(X_tX + lambd_I)
    w = part_1.dot(X_t).dot(y)                                          
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # mse
    # IMPLEMENT THIS METHOD
    Xtest_w = Xtest.dot(w)  #N x 1
    yi_minus_Xtest_w = ytest - Xtest_w #N x 1
    mse = np.sum(np.square(yi_minus_Xtest_w)) / yi_minus_Xtest_w.shape[0]
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    
    w = np.asmatrix(w).T
    y_Xw = y - X.dot(w)
    part_1 = 0.5 * np.dot(y_Xw.T, y_Xw)
    part_2 = 0.5 * lambd * np.dot(w.T, w)
    error = part_1 + part_2
    error_grad = -X.T.dot(y - X.dot(w)) + lambd * w  
    error_grad = np.squeeze(np.array(error_grad))
    
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    
    # IMPLEMENT THIS METHOD
    Xp = np.empty((len(x), 0),float)
    for i in range(p+1):
        x_elem = np.array([x ** i]).T
        Xp = np.append(Xp, x_elem, axis=1)
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
ytest = np.reshape(ytest, (100, ))
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()

## get optimal value of lambd
mses3_train_min_index = np.argmin(mses3_train)
mses3_train_min = mses3_train[mses3_train_min_index]
mses3_train_lambd = mses3_train_min_index * 0.01
mses3_min_index = np.argmin(mses3)
mses3_min = mses3[mses3_min_index]
mses3_lambd = mses3_min_index * 0.01
'''
print("minimum mse for train data: ", mses3_train_min)
print("optimal value of lambd for train data: ", mses3_train_lambd)
print("minimum mse for test data: ", mses3_min)
print("optimal value of lambd for test data: ", mses3_lambd)

w_test_i = learnOLERegression(Xtest_i,ytest)
w_ridge_test_i = learnRidgeRegression(Xtest_i,ytest,mses3_lambd) 
# comparing the two for sparsity
w1 = np.squeeze(np.asarray(w_test_i))
w2 = np.squeeze(np.asarray(w_ridge_test_i))
fig = plt.figure(figsize=[20,6])
plt.title('magnitudes of weights for test data')
plt.bar(range(1,len(w1)+1),w1,color='red',width=0.4,alpha=0.6)
plt.bar(np.arange(1.4,len(w2)+1),w2,color='green',width=0.4,alpha=0.6)
plt.legend(['Linear Regression', 'Ridge'])
'''
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()

#compare P4 with P3
mses4_train_min_index = np.argmin(mses4_train)
mses4_train_min = mses4_train[mses4_train_min_index]
mses4_train_lambd = mses4_train_min_index * 0.01
mses4_min_index = np.argmin(mses4)
mses4_min = mses4[mses4_min_index]
mses4_lambd = mses4_min_index * 0.01
'''
print("P3 minimum mse for train data: ", mses3_train_min)
print("P3 optimal value of lambd for train data: ", mses3_train_lambd)
print("P3 minimum mse for test data: ", mses3_min)
print("P3 optimal value of lambd for test data: ", mses3_lambd)
print()
print("P4 minimum mse for train data: ", mses4_train_min)
print("P4 optimal value of lambd for train data: ", mses4_train_lambd)
print("P4 minimum mse for test data: ", mses4_min)
print("P4 optimal value of lambd for test data: ", mses4_lambd)
'''
# Problem 5
pmax = 7
lambda_opt = mses3_lambd # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
'''
print('lambda_opt estimated from Problem 3: ', mses3_lambd)
print("optimal value of p for train data")
print('No Regularizaion: ',np.argmin(mses5_train.T[0]) +1 )
print('Regularizaion: ',np.argmin(mses5_train.T[1]) +1 )
print("optimal value of p for test data")
print('No Regularizaion: ', np.argmin(mses5.T[0]) +1 )
print('Regularizaion: ',np.argmin(mses5.T[1]) +1 )
'''
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
