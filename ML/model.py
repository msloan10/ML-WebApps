import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
import pickle

#read data
mydata = pd.read_csv('Telco_Data.csv')

#encodecatVar: encode var. for algo
def encodeCatVar(df):
    binCols = list(df.select_dtypes(include=['object']))
    for col in binCols:
        df[col] = df[col].astype('category')
        df[col + '_cat'] = df[col].cat.codes
    return df

#clean data
mydata = mydata.drop('customerID', 1)
mydata.drop(mydata[mydata['TotalCharges'] == ' '].index, inplace=True)
mydata['TotalCharges'] = mydata['TotalCharges'].astype(float)

# make copy; continue cleaning
df = mydata.copy()
paymentMethods = df.PaymentMethod.unique()
df['PaymentMethod'] = df["PaymentMethod"].replace(paymentMethods, [0,0,1,1])
df = encodeCatVar(df).copy()

#drop uneeded columns

Y = df['Churn_cat']

cat_drop = list(df.select_dtypes(include = ['category']))

drop_cols = ['gender_cat', 'PhoneService_cat', 'MultipleLines_cat', 'InternetService_cat','StreamingTV_cat','StreamingMovies_cat',
 'Churn_cat','TotalCharges','SeniorCitizen','Dependents_cat' ]


drop_cols = cat_drop+ drop_cols

dfinal = df.copy()
dfinal = dfinal.drop(drop_cols, 1)

#test, train, split
X_train, X_test, y_train, y_test = train_test_split(dfinal, Y,test_size = .2,shuffle = False)

#make model 
model = LogisticRegression(solver='lbfgs',max_iter=200)
model.fit(X_train, y_train)

#testing accuracy/MSE
y_predicted = model.predict(X_test)
print("Model Accuracy: ",model.score(X_test,y_test))
print("MSE: ", mean_squared_error(y_test, y_predicted))


                                                