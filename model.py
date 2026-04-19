import pandas 
import numpy
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error , r2_score
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
df['furnishingstatus'] = df['furnishingstatus'].map(mapping2)

x = df[['area'],['bedrooms'],['bathrooms'],['stories'],['mainroad'],['guestroom'],['basement'],['hotwaterheating'],['airconditioning'],['parking'],['prefarea'],['furnishingstatus']]
y = df['price']
x_train , x_test , y_train , y_test = train_test_split(x , y , train_size=0.2 , random_state=42)

model = LinearRegression()
model.fit()