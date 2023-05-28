# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

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
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK RAJ.M
RegisterNumber:  212221040073
*/
```


```
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')
print(dataset)

# assigning hours to X & Scores to Y
X=dataset.iloc[:,:1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title('Training set (H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show

plt.scatter(X_test,Y_test,color='purple')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse
```






 Output:
 
 
 ![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/542a80f6-5067-47e3-8bb2-2506fa786759)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/05eb106a-68ae-4c05-9df1-89fc16ae4cff)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/1fe27c6a-599c-4239-bb5e-637594336730)



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/970ab55a-2b72-4011-9730-a4d09ab42f42)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/b7b7093b-8732-4883-8679-bf0bdf83f4ab)

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/64ef5c39-873d-4f26-a7a4-181fec67404a)


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/128134963/2cf35d76-8f53-4e6e-9c82-eeaa8b3ee850)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
