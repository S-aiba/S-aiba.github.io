import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets,linear_model
from sklearn.linear_model import LinearRegression

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_data.csv',dtype= dtype_dict )
train_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_train_data.csv', dtype=dtype_dict)
test_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_test_data.csv',dtype=dtype_dict)
valid_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_valid_data.csv',dtype=dtype_dict)

set1=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_set_1_data.csv',dtype=dtype_dict)
set2=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_set_2_data.csv',dtype=dtype_dict)
set3=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_set_3_data.csv',dtype=dtype_dict)
set4=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_set_4_data.csv',dtype=dtype_dict)

def polynomial_sframe(feature, degree):
    poly_frame=pd.DataFrame()
    poly_frame['power_1']=feature
    if(degree>1):
        for i in range(2,degree+1):
            name='power_'+str(i)
            poly_frame[name]=feature.apply(lambda x:x**i)
    return poly_frame

sales = sales.sort(['sqft_living','price'])
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price']

model1 = LinearRegression()
model1.fit(poly1_data['power_1'].reshape(len(poly1_data['power_1']),1), poly1_data['price'].reshape(len(poly1_data['price']),1))
print (model1.intercept_,model1.coef_)

plt.plot(poly1_data['power_1'].reshape(len(poly1_data['power_1']),1),poly1_data['price'].reshape(len(poly1_data['price']),1),'.',
poly1_data['power_1'].reshape(len(poly1_data['power_1']),1), model1.predict(poly1_data['power_1'].reshape(len(poly1_data['power_1']),1)),'-')


poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features_2 = poly2_data.columns
poly2_data['price'] = sales['price']

model2 = LinearRegression()
model2.fit(poly2_data[my_features_2], poly2_data['price'])

print (model2.intercept_,model2.coef_)

plt.plot(poly2_data['power_1'].reshape(len(poly2_data['power_1']),1),poly2_data['price'].reshape(len(poly2_data['price']),1),'.',
poly2_data['power_1'].reshape(len(poly2_data['power_1']),1), model2.predict(poly2_data[my_features_2]),'-')

poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features_3 = poly3_data.columns
poly3_data['price'] = sales['price']

model3 = LinearRegression()
model3.fit(poly3_data[my_features_3], poly3_data['price'])

print (model3.intercept_,model3.coef_)

plt.plot(poly3_data['power_1'].reshape(len(poly3_data['power_1']),1),poly3_data['price'].reshape(len(poly3_data['price']),1),'.',
poly3_data['power_1'].reshape(len(poly3_data['power_1']),1), model3.predict(poly3_data[my_features_3]),'-')

poly15_data=polynomial_sframe(sales['sqft_living'],15)
# print (poly15_data.head())
my_features_15=poly15_data.columns
poly15_data['price']=sales['price']

model15=LinearRegression()
model15.fit(poly15_data[my_features_15],poly15_data['price'])
print (model15.intercept_,model15.coef_)

plt.plot(poly15_data['power_1'].reshape(len(poly15_data['power_1']),1),poly15_data['price'].reshape(len(poly15_data['price']),1),'.',
poly15_data['power_1'].reshape(len(poly15_data['power_1']),1), model15.predict(poly15_data[my_features_15]),'-')

def plot_15degree(set):
    poly15_data = polynomial_sframe(set['sqft_living'], 15)
    my_features = poly15_data.columns
    poly15_data['price'] = set['price']
    model15 = LinearRegression()
    model15.fit(poly15_data[my_features], poly15_data['price'])
    # print model15.intercept_, model15.coef_ #use print vs. return since return can only apply to function
    plt.plot(poly15_data['power_1'].reshape(len(poly15_data['power_1']),1),poly15_data['price'].reshape(len(poly15_data['price']),1),'.',
             poly15_data['power_1'].reshape(len(poly15_data['power_1']),1), model15.predict(poly15_data[my_features]),'-')

plot_15degree(set1)
plot_15degree(set2)
plot_15degree(set3)
plot_15degree(set4)

def plot_degree(data1, data2, degree):
    poly_data=polynomial_sframe(data1['sqft_living'],degree)
    valid_datas=polynomial_sframe(data2['sqft_living'],degree)
    my_features=poly_data.columns
    my_features_valid=valid_datas.columns
    poly_data['price']=data1['price']
    valid_datas['price']=data2['price']
    model=LinearRegression()
    model.fit(poly_data[my_features],poly_data['price'])
    return (np.sum((model.predict(valid_datas[my_features_valid])-valid_datas['price'])**2))

range_list=[]
for i in range(1,16):
    range_list.append((plot_degree(train_data,valid_data,i),i))

print (min(range_list))

print(plot_degree(train_data,test_data,6))