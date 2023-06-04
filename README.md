# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to implement the simple linear regression model for predicting the marks scored.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
Step 1:
Use the standard libraries such as numpy, pandas, matplotlib.pyplot in python for the simple linear regression model for predicting the marks scored.

Step 2:
Set variables for assigning dataset values and implement the .iloc module for slicing the values of the variables X and y.

Step 3:
Import the following modules for linear regression; from sklearn.model_selection import train_test_split and also from sklearn.linear_model import LinearRegression.

Step 4:
Assign the points for representing the points required for the fitting of the straight line in the graph.

Step 5:
Predict the regression of the straight line for marks by using the representation of the graph.

Step 6:
Compare the graphs (Training set, Testing set) and hence we obtained the simple linear regression model for predicting the marks scored using the given datas.

Step 7:
End the program.
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: KARTHICK RAJ M
RegisterNumber:212221040073
*/
```import pandas as pd
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

## Output:

![image](https://user-images.githubusercontent.com/94911373/228476761-420b67d9-2a7d-448b-b27e-4f1804f7fd8d.png)


![image](https://user-images.githubusercontent.com/94911373/228476846-f07ce461-e8b7-41a4-a148-d5fde1e7ee6e.png)


![image](https://user-images.githubusercontent.com/94911373/228476904-ad0fb962-26f6-45c3-9682-275f9c75080c.png)



![image](https://user-images.githubusercontent.com/94911373/228476956-92c91ad6-5321-43a6-9b0e-4e6153121913.png)



![image](https://user-images.githubusercontent.com/94911373/228477010-1176c915-bd6c-44a7-9ce9-b55c852ae7d6.png)





![image](https://user-images.githubusercontent.com/94911373/228477085-f182b942-8344-4b00-ba1d-24c5a18de642.png)





![image](https://user-images.githubusercontent.com/94911373/228477174-8c2a5a6e-40a0-4a62-8996-38afddb1c07e.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
