import pandas 
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , root_mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

df = pandas.read_csv("Housing.csv")

mapping = {'yes' : 1 , 'no' : 0} 
df['mainroad'] = df['mainroad'].map(mapping)
df['guestroom'] = df['guestroom'].map(mapping)
df['basement'] = df['basement'].map(mapping)
df['hotwaterheating'] = df['hotwaterheating'].map(mapping)
df['airconditioning'] = df['airconditioning'].map(mapping)
df['prefarea'] = df['prefarea'].map(mapping)

mapping2={'furnished' : 2 , 'semi-furnished' : 1 , 'unfurnished' : 0}