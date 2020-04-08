import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt
from math import sqrt
from sklearn import linear_model
from sklearn.linear_model import Lasso

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_data.csv',dtype= dtype_dict )
training=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_train_data.csv', dtype=dtype_dict)
testing=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_test_data.csv',dtype=dtype_dict)
validation=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_valid_data.csv',dtype=dtype_dict)

sales['floors']=sales['floors'].astype(float)
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(lambda x:sqrt(x))
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(lambda x:sqrt(x))
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

model_all = Lasso(alpha=5e2, normalize=True)
model_all.fit(sales[all_features], sales['price'])

print (pd.Series(model_all.coef_,index=all_features))

testing['floors']=testing['floors'].astype(float)
training['floors']=training['floors'].astype(float)
validation['floors']=validation['floors'].astype(float)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']

l1_penalty=np.logspace(1,7,num=13)

val_err={}
for i in l1_penalty:
    model=Lasso(alpha=i,normalize=True)
    model.fit(training[all_features],training['price'])
    val_err[i]=np.sum((model.predict(validation[all_features])-validation['price'])**2)
    print i,val_err[i]
print(min(val_err.items(),key=lambda x:x[1]))

model_train=Lasso(alpha=10.0,normalize=True)
model_train.fit(training[all_features],training['price'])

rss_test=np.sum((model_train.predict(testing[all_features])-testing['price'])**2)
print rss_test

non_zeroes=np.count_nonzero(model_train.coef_)+np.count_nonzero(model_train.intercept_)
print non_zeroes

max_nonzeros=7
non0=[]
l1_penalty=np.logspace(1,4,num=20)
for i in l1_penalty:
    model=Lasso(alpha=i,normalize=True)
    model.fit(training[all_features],training['price'])
    if(model.intercept_<>0):
        count=np.count_nonzero(model.coef_)+np.count_nonzero(model.intercept_)
    else:
        count=np.count_nonzero(model.coef_)+1
    non0.append((i,count))

print non0

list_greater_l1=[k for k,v in non0 if v>max_nonzeros]
l1_penalty_min=max(list_greater_l1)
print l1_penalty_min

list_smaller_l1=[k for k,v in non0 if v<max_nonzeros]
l1_penalty_max=min(list_smaller_l1)
print l1_penalty_max

l1_penalty=np.linspace(l1_penalty_min,l1_penalty_max,num=20)
non0_2=[]
rss=[]
for i in l1_penalty:
    model = Lasso(alpha=i, normalize=True)
    model.fit(training[all_features], training['price'])
    rss_val= np.sum((model.predict(validation[all_features]) - validation['price']) ** 2)
    rss.append((i,rss_val))
    if(model.intercept_<>0):
        count=np.count_nonzero(model.coef_)+np.count_nonzero(model.intercept_)
    else:
        count=np.count_nonzero(model.coef_)+1
    non0_2.append((i,count))

print non0_2
max_non0=[k for k,v in non0_2 if v==max_nonzeros]
print max_non0

rss_min=[(v,k)for k,v in rss if k in max_non0 ]
rss_min.sort()

min_l1=rss_min[0][1]
print min_l1

train_model=Lasso(alpha=min_l1,normalize=True)
train_model.fit(training[all_features],training['price'])

print (pd.Series(train_model.coef_,index=all_features))

predictions=train_model.predict(testing[all_features])
print predictions[0]
print testing['price'][0]











