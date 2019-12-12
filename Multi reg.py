import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv("C:/Users/Arif/Downloads/Data/50_Startups.csv")
X=data.iloc[:,:-1].values
y=data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

import statsmodels.regression.linear_model as sm
X=np.append(arr=np.ones((50,1)).astype(int),values=X,axis=1)

xopt=X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=xopt).fit()
regressor_OLS.summary()