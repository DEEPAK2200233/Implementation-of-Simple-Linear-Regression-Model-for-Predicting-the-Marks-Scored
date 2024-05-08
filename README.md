# Ex.No.2-Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DEEPAK RAJ S
RegisterNumber:212222240023
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse
df=pd.read_csv("student_scores.csv")
df.head()
df.tail()
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,regressor.predict(x_train),color='yellow')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/0c7a7db6-866c-4413-9dcd-72d09bef4131)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/71bb107c-3683-495a-9cd3-493328d71add)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/dd3d3059-f86a-4488-860e-aa0180cf27de)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/c7d35db4-a5ea-4ea3-9b58-162978519836)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/45d1165a-12c0-4c26-b719-3acbf5cf854e)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/9e5f72a1-d709-4538-bcb4-3ddecd11bb9c)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/de6891c6-1a6c-43f0-a958-7347537a1029)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/bceb01ee-b4b6-48bb-9593-112619b53e57)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/09455499-eccf-4086-ad9c-b0bbe0577d4d)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/08d45b8b-f317-41c0-9989-88795a73ad50)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/2fe9b461-df07-403a-abff-6544d68ba2c8)





## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
