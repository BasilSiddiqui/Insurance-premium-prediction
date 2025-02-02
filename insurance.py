import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#Loading the data set
insurance_dataset = pd.read_csv(r"C:\Users\basil\OneDrive\Desktop\Base\Other\Datasets\insurance.csv")

#First 5 rows
insurance_dataset.head()

#Number of rows and columns
insurance_dataset.shape

#Basic information of dataset
insurance_dataset.info()

#Categorical featues: 
    #sex
    #smoker
    #region
'''
#Checking for missing values
insurance_dataset.isnull().sum()

#Statistical measures of dataset
insurance_dataset.describe()

#Distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age distribution')
plt.show()

#Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex',data=insurance_dataset)
plt.title('Sex distribution')

#Value count for sex
insurance_dataset['sex'].value_counts()

#BMI distributions
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['bmi'])
plt.title('BMI distribution')
plt.show()

#Children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()

insurance_dataset['children'].value_counts()

#Smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title("Smokers")
plt.show()

insurance_dataset['smoker'].value_counts()

#Region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title("Regions")
plt.show()

insurance_dataset['region'].value_counts()

#Charges distributions
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['charges'])
plt.title('Charges distribution')
plt.show()

insurance_dataset['charges'].mean()
'''
#Encoding colums
insurance_dataset.replace({'sex':{"male":0,"female":1}}, inplace=True)
insurance_dataset.replace({'smoker':{"yes":0,"no":1}}, inplace=True)
insurance_dataset.replace({'region':{"southeast":0,"southwest":1,"northeast":2,"northwest":3}}, inplace=True)

#Splitting data into features and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

#Splitting into training data and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Model training
#Loading the Linear Regression model
regressor = LinearRegression()

regressor.fit(X_train,Y_train)

#Model evaluation
#Prediction on training data
training_data_prediction = regressor.predict(X_train)

#R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print("R squared value: ",r2_train)

#Prediction on testing data
testing_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, testing_data_prediction)

#Building a Predictive System
input_data = (56,0,40.3,0,1,1)

#changing input_data into numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print("Prediction is:",prediction[0])  
