import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics

df = pd.read_csv('customer_churn.csv')

df1 = df.drop('customerID' , axis='columns')
#print(df1.dtypes)
#print(df1[pd.to_numeric(df1['TotalCharges'] , errors='coerce').isnull()].shape )
df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'] , errors='coerce')

df2 = df1[~df1['TotalCharges'].isnull()]

df2 = df2.replace('No phone service' , 'No')
df2 = df2.replace('No internet service' , 'No')

yes_no_columns = ['Partner' , 'Dependents' , 'PhoneService' , 'MultipleLines' , 'OnlineSecurity' ,
                  'OnlineBackup' , 'DeviceProtection' , 'TechSupport' , 'StreamingTV' , 'StreamingMovies' , 
                  'PaperlessBilling' ,
                  'Churn']

for col in yes_no_columns:
  df2[col].replace({'Yes' :1 , 'No':0} , inplace=True)

df2['gender'].replace({'Female':1 ,  'Male':0} , inplace=True)

df3 = pd.get_dummies(data = df2 , columns = ['InternetService' , 'Contract' , 'PaymentMethod'])

X = df3.drop('Churn' , axis='columns')
Y = df3['Churn']

xtrain,xtest,ytrain,ytest = train_test_split(X,Y,random_state=0,train_size=0.8)

lmodel = LogisticRegression()
lmodel.fit(xtrain,ytrain)

# print('Training Accuarcy',lmodel.score(xtrain,ytrain))
# print('Testing Accuarcy',lmodel.score(xtest,ytest))
#print(xtrain.head())

col_scale = ['tenure' , 'MonthlyCharges' ,	'TotalCharges']

scaler = MinMaxScaler()

xtrain[col_scale] = scaler.fit_transform(xtrain[col_scale])
xtest[col_scale] = scaler.fit_transform(xtest[col_scale])

model = models.Sequential()                                      
model.add(layers.Dense(16 , activation = 'relu' , input_dim = xtrain.shape[1]))           
model.add(layers.Dense(1 , activation = 'sigmoid'))   

model.compile(optimizer = 'sgd',loss = 'binary_crossentropy' ,metrics = ['accuracy'])

model.fit(xtrain,ytrain,epochs=30)

print(model.evaluate(xtest,ytest))
#print(model.predict(xtrain))
print(model.predict(xtest))