import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"D:\Data Science\16th,17th\16th,17th\1.POLYNOMIAL REGRESSION\emp_sal.csv")


X = dataset.iloc[:, 1:2].values

y = dataset.iloc[:, 2].values


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_reg.predict([[6.5]])

lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


from sklearn.svm import SVR
regressor = SVR(kernel='precomputed',degree=4)
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=3, algorithm = 'kd_tree')
regressor.fit(X, y)
y_pred = regressor.predict([[6.5]])
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor(max_depth = 6, splitter = 'random',criterion = 'friedman_mse', random_state = 0)
regressor_dtr.fit(X,y)
y_pred_dtr = regressor_dtr.predict([[6.5]])

from sklearn.ensemble import RandomForestRegressor 
reg = RandomForestRegressor(n_estimators = 300, random_state = 0)
reg.fit(X,y)
y_pred = reg.predict([[6.5]])
plt.scatter(X, y, color = 'red')
plt.plot(X,regressor.predict(X), color = 'blue')
plt.title('Truth or bluff (Decision tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()















