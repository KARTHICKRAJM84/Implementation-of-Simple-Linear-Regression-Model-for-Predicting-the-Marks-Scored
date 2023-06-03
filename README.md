# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6.Obtain the graph.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KARTHICK RAJ.M
RegisterNumber:  212221040073
*/
```


```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
```






 Output:
 
 
 Array Value of x:
 
![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/51f96a28-27ba-48e2-9749-6c812561237f)



Array Value of Y:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/5bdf2d81-3aa0-4e37-9265-bfb8ca922633)


Exam 1 - score graph:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/736e3b55-5573-4db6-b52e-41eabaa441a1)


Sigmoid function graph:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/83f21b74-0237-4e4b-93d4-bb4a6176d332)

x_train_grad value:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/10f173f2-c7a7-4f8b-8885-673b7fc37698)


y_train_grad value:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/697f4632-71c0-4ae7-97b3-7fb88bfcfc72)


Print res.x:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/cecb2fa7-9b35-47d4-b856-b12f48c48160)


Decision boundary - graph for exam score:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/f4b2a432-95b2-4329-a1c4-220179648b0b)


Proability value:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/ba892303-4251-4e5d-a879-d2de143fb23d)


Prediction value of mean:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/1c271770-d713-4ecf-8b53-57c8d08da2e7)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
