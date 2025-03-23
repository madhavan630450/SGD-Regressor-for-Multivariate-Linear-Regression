# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the California housing dataset, create a DataFrame, and separate features and target variables.

 2. Split the data into training and testing sets, then standardize the features and target variables.

 3. Initialize and train a MultiOutputRegressor with an SGDRegressor on the training data.

 4. Predict on the test set, reverse scaling, and calculate the Mean Squared Error (MSE).


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MARIMUTHU MATHAVAN
RegisterNumber:  212224230153
*/
```
```
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())
X=df.info()
X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()
Y=df[['AveOccup','HousingPrice']]
Y.info()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
print(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
print(Y_test)
Y_mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",Y_mse)
```

## Output:
![Screenshot 2025-03-23 140523](https://github.com/user-attachments/assets/b0c08cd3-c0d3-465f-834c-2e1147fd45db)
![Screenshot 2025-03-23 140607](https://github.com/user-attachments/assets/0ba18a24-ff46-49b7-b1f1-75aa8f6b9feb)
![Screenshot 2025-03-23 140631](https://github.com/user-attachments/assets/3159bf66-89cc-49dd-91fc-2b12862e1ae1)
![Screenshot 2025-03-23 140655](https://github.com/user-attachments/assets/0c6698ae-ed00-4fba-956b-c6b62678a509)
![Screenshot 2025-03-23 140731](https://github.com/user-attachments/assets/8aa65a37-11f7-491f-b1f4-556db2cf7c25)
![Screenshot 2025-03-23 140809](https://github.com/user-attachments/assets/c293e9a2-9214-4ffe-a793-57b8e2dbc6a2)
![Screenshot 2025-03-23 140834](https://github.com/user-attachments/assets/4c52c23b-3228-4550-91ca-79adc8433801)
![Screenshot 2025-03-23 140908](https://github.com/user-attachments/assets/ba17e70f-ba05-4da3-a830-f1d1e9424582)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
