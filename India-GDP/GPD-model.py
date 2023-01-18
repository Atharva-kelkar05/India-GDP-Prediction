import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

#loading csv
data = pd.read_csv('E:\Project_Exhi-2\India-GDP\dataset.csv')

data.columns = (["serial","year","gdp","percap","growth"])
#data.info()

data.serial = data.serial.astype(float)
data.year =data.year.astype(int)
data.gdp =data.gdp.astype(float)
data.percap =data.percap.astype(float)
data.growth =data.growth.astype(float)

#data.describe()

#print(data.notnull().sum())

#Visualizing the available dataset and previous trends of GDP
# fig=plt.figure(figsize=(12,4))
# data.groupby('year')['gdp'].mean().sort_values().plot(kind='bar', color='coral')
# plt.title('GDP of India from 1960-2022')
# plt.xlabel("Year")
# plt.ylabel("GDP in the FY")
# plt.show()

#Region Transform
data_final= pd.concat([data,pd.get_dummies(data['year'], prefix='year')],axis=1).drop(['year'],axis=1)

#data split-1: all of out final dataset, with no scaling.

y=data_final['gdp']
X=data_final.drop(['gdp','percap'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=101)

#Model Training-

# Random Forest -
rf1=RandomForestRegressor(random_state=101, n_estimators=200)


rf1.fit(X_train, y_train)

#Predicting using predict() function;
rf1_pred= rf1.predict(X_test)

# Printing the results after evaluation:
print("\nRandom Forest Performance:")
print("\nAll features, no scaling:")
print("MAE:",metrics.mean_absolute_error(y_test,rf1_pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, rf1_pred)))
print("R2_Score: ", metrics.r2_score(y_test, rf1_pred))