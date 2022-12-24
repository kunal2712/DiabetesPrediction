import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model

df = pd.read_csv('diabetes_data.csv')
#df.rename(columns={'diabetes': 'diabete'}, inplace=True)
y = df['diabetes']
X = df.drop(columns=['diabetes'])

#print(df.columns)

lm = linear_model.LinearRegression()
lm.fit(X.values, y)
key = lm.predict([[5, 137,85,15 , 22.3, 43.1, 2.288, 22]])
print(key)
#pickle.dump(lm, open('model.pkl','wb')) 
#if key >= 0.5: print("Patient have Diabetes")
#else: print("Patient not have Diabetes")
