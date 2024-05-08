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
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/d06991b1-eec3-4ba3-88d4-b7dba801e810)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/309cf8a1-1f74-4f06-9857-6c8d6b158f8c)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/73310fcd-9fea-41ff-8e4b-fc4f1c69bd6c)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/8adc0b42-cfde-4ec0-88fb-3619cd024cb2)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/3c4b3f37-cd48-4f72-b651-7c5d4fd042c7)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/0c1c58ae-5167-4319-a43f-72d67ef799f4)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/093c6362-7a77-4575-8200-a8a94d3013b8)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/be155446-b6e5-438f-9a54-15e84471b78e)
![image](https://github.com/DEEPAK2200233/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707676/c0405ed3-2613-4a7c-a3a6-663a0664b268)






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
