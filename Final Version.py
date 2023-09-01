import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import gradio  as gr
inputs = [gr.Dataframe(row_count = (5, "dynamic"), col_count=(8,"dynamic"), label="Input Data", interactive=1)]
outputs = [gr.Dataframe(row_count = (5, "dynamic"), col_count=(9, "fixed"), label="Predictions", headers=["Number","Transaction Date","House Age",
            "Distance from MTR Station","Number of Convinience Stores Nearby","Latitude","Longitude","Correct Value","Validity"])]

# importing data
df = pd.read_csv('C:/Users/Admin/Downloads/Real estate.csv')
df.drop('No', inplace=True, axis=1)

print(df.head())
  
print(df.columns)
  
# plotting a scatterplot
sns.scatterplot(x='X2 house age',
                y='Y house price of unit area',  data=df)
  
# creating feature variables
X = df.drop('Y house price of unit area', axis=1)
y = df['Y house price of unit area']
  
print(X)
print(y)
  
# creating train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101)
  
# creating a regression model
model = LinearRegression()

# fitting the model
model.fit(X_train, y_train)
  
# making predictions
predictions = model.predict(X_test)

# model evaluation
print('mean_squared_error : ', mean_squared_error(y_test, predictions))
print('mean_absolute_error : ', mean_absolute_error(y_test, predictions))


df2=pd.read_csv('C:/Users/Admin/Desktop/Inputs.csv')
df2.drop('No',inplace=True,axis=1)
x1=df2.drop('Y house price of unit area',axis=1)
y1=df2['Y house price of unit area']
f=list(model.predict(x1))
l=[]
for i,j in zip(y1,f):
    if(((float(i)-float(j))*100/float(j))>150):
        l.append("Fraudulent")
    else:
        l.append("Valid")
df3=df2.copy(deep=True)        
df3['X4 Correct Price']=pd.Series(predictions)
df3['X5 Validity']=pd.Series(l)
df3.to_csv("C:/Users/Admin/Desktop/Outputs.csv")
def dis(df3):
    return df3
gr.Interface(fn = dis, inputs = inputs, outputs = outputs, examples = [[df3.head(10)]]).launch()