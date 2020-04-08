import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from math import sqrt
from math import log

sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
train_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_train_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_test_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})


def simple_linear_regression(input_feature, output):
    numerator=(input_feature*output).mean(axis=0)-(output.mean(axis=0)*input_feature.mean(axis=0))
    denominator=(input_feature*input_feature).mean(axis=0)-input_feature.mean(axis=0)*input_feature.mean(axis=0)
    slope=numerator/denominator
    intercept=output.mean(axis=0)-slope*input_feature.mean(axis=0)
    return (intercept,slope)

sqft_living_list=[i for i in train_data['sqft_living']]
sqft_living_array=np.array(sqft_living_list)

price_list=[i for i in train_data['price']]
price_array=np.array(price_list)

def get_regression_predictions(input_feature, intercept, slope):
    output=input_feature*slope+intercept
    return output

def get_residual_sum_of_squares(input_feature, output, intercept,slope):
    prediction=input_feature*slope+intercept
    residuals=prediction-output
    rss=(residuals**2).sum()
    return rss

def inverse_regression_predictions(output, intercept, slope):
    estimated_input=(output-intercept)/slope
    return estimated_input

intercept_train,slope_train=simple_linear_regression(sqft_living_array,price_array)

input_feature=2650
print(get_regression_predictions(input_feature,intercept_train,slope_train))

print(get_residual_sum_of_squares(sqft_living_array,price_array,intercept_train,slope_train))

output_price=800000
print(inverse_regression_predictions(output_price,intercept_train,slope_train))

sqft_test=[i for i in test_data['sqft_living']]
sqft_test_array=np.array(sqft_test)

price_test=[i for i in test_data['price']]
price_test_array=np.array(price_test)

bedroom_test=[i for i in test_data['bedrooms']]
bedroom_test_array=np.array(bedroom_test)

intercept_test_sqft,slope_test_sqft=simple_linear_regression(sqft_test_array,price_test_array)
print(get_residual_sum_of_squares(sqft_test_array,price_test_array,intercept_test_sqft,slope_test_sqft))

intercept_test_bed,slope_test_bed=simple_linear_regression(bedroom_test_array,price_test_array)
print (get_residual_sum_of_squares(bedroom_test_array,price_test_array,intercept_test_bed,slope_test_bed))

print(' ')
print('WEEK2')

train_data['bedrooms_squared']=train_data['bedrooms'].apply(lambda x:x**2)
train_data['bed_bath_rooms']=train_data['bedrooms']*train_data['bathrooms']
train_data['log_sqft_living']=train_data['sqft_living'].apply(lambda x:log(x))
train_data['lat_plus_long']=train_data['lat']+train_data['long']

test_data['bedrooms_squared']=test_data['bedrooms'].apply(lambda x:x**2)
test_data['bed_bath_rooms']=test_data['bedrooms']*test_data['bathrooms']
test_data['log_sqft_living']=test_data['sqft_living'].apply(lambda x:log(x))
test_data['lat_plus_long']=test_data['lat']+test_data['long']

print(test_data['bedrooms_squared'].mean())
print(test_data['bed_bath_rooms'].mean())
print (test_data['log_sqft_living'].mean())
print (test_data['lat_plus_long'].mean())

df_mod1=pd.DataFrame(train_data,columns=['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'])
df_mod2=pd.DataFrame(train_data,columns=['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms'])
df_mod3=pd.DataFrame(train_data,columns=['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long'])

output=pd.DataFrame(train_data,columns=['price'])
model1=LinearRegression()
model1.fit(df_mod1,output)
model2=LinearRegression()
model2.fit(df_mod2,output)
model3=LinearRegression()
model3.fit(df_mod3,output)

print(model1.coef_)
print (model2.coef_)

print(np.sum(((model1.predict(df_mod1[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]))-output)**2))

print(np.sum(((model2.predict(df_mod2[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']]))-output)**2))

print(np.sum(((model3.predict(df_mod3[['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]))-output)**2))


df_mod1_test=pd.DataFrame(test_data,columns=['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long'])
df_mod2_test=pd.DataFrame(test_data,columns=['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms'])
df_mod3_test=pd.DataFrame(test_data,columns=['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long'])
output_test=pd.DataFrame(test_data,columns=['price'])

print(np.sum(((model1.predict(df_mod1_test[['sqft_living', 'bedrooms', 'bathrooms', 'lat', 'long']]))-output_test)**2))

print(np.sum(((model2.predict(df_mod2_test[['sqft_living','bedrooms','bathrooms','lat','long','bed_bath_rooms']]))-output_test)**2))

print(np.sum(((model3.predict(df_mod3_test[['sqft_living', 'bedrooms', 'bathrooms', 'lat','long', 'bed_bath_rooms', 'bedrooms_squared', 'log_sqft_living', 'lat_plus_long']]))-output_test)**2))
