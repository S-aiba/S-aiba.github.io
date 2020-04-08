import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt
from math import sqrt
from sklearn import datasets,linear_model
from sklearn.linear_model import Ridge

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

l2_small_penalty = 1.5e-5

poly15_data=polynomial_sframe(sales['sqft_living'],15)
features_15=poly15_data.columns
poly15_data['price']=sales['price']

model=Ridge(alpha=l2_small_penalty,normalize=True)
model.fit(poly15_data[features_15],poly15_data['price'])

print (model.intercept_,model.coef_)

def plot_15degree(set,l2_penalty):
    poly15_data = polynomial_sframe(set['sqft_living'], 15)
    my_features = poly15_data.columns
    poly15_data['price'] = set['price']
    model15 = Ridge(alpha=l2_penalty,normalize=True)
    model15.fit(poly15_data[my_features], poly15_data['price'])
    print pd.Series(model15.coef_,index=my_features)
    plt.plot(poly15_data['power_1'].reshape(len(poly15_data['power_1']),1),poly15_data['price'].reshape(len(poly15_data['price']),1),'.',
             poly15_data['power_1'].reshape(len(poly15_data['power_1']),1), model15.predict(poly15_data[my_features]),'-')

l2_small_penalty=1e-9
plot_15degree(set1,l2_small_penalty)
plot_15degree(set2,l2_small_penalty)
plot_15degree(set3,l2_small_penalty)
plot_15degree(set4,l2_small_penalty)

l2_large_penalty=1.23e2

plot_15degree(set1,l2_large_penalty)
plot_15degree(set2,l2_large_penalty)
plot_15degree(set3,l2_large_penalty)
plot_15degree(set4,l2_large_penalty)

train_valid_shuffled=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_train_valid_shuffled.csv',dtype=dtype_dict)

n=len(train_valid_shuffled)
k=10
for i in xrange(k):
    start=(n*i)/k
    end=(n*(i+1))/k-1
    print (i,start,end)

print (train_valid_shuffled[0:10])

def k_fold_cross_validation(k, l2_penalty, data):
    n=len(data)
    rss_sum=0
    for i in xrange(k):
        start=(n*i)/k
        end=(n*(i+1))/k-1
        validation_data=data[start:end+1]
        training_data=data[0:start].append(data[end+1:n])

        train_frame=polynomial_sframe(training_data['sqft_living'],15)
        valid_frame=polynomial_sframe(validation_data['sqft_living'],15)
        features_train=train_frame.columns
        features_valid=valid_frame.columns

        train_frame['price']=training_data['price']
        valid_frame['price']=validation_data['price']

        model=Ridge(alpha=l2_penalty,normalize=True)
        model.fit(train_frame[features_train],train_frame['price'])

        predictions=model.predict(valid_frame[features_valid])
        rss=np.sum((predictions-valid_frame['price'])**2)
        rss_sum+=rss
    return rss_sum/k

l2_penalty=np.logspace(3,9,num=13)

val_err={}
for i in l2_penalty:
    val_err[i]=k_fold_cross_validation(10,i,train_valid_shuffled)
    print i,k_fold_cross_validation(10,i,train_valid_shuffled)

print(min(val_err.items(),key=lambda x:x[1]))

poly_15_data=polynomial_sframe(train_valid_shuffled['sqft_living'],15)
features_15_tvs=poly_15_data.columns
poly_15_data['price']=train_valid_shuffled['price']
l2_penalty=1000
model15tvs=Ridge(alpha=l2_penalty,normalize=True)
model15tvs.fit(poly_15_data[features_15_tvs],poly_15_data['price'])

poly_15_test=polynomial_sframe(test_data['sqft_living'],15)
features_test_15=poly_15_test.columns
poly_15_test['price']=test_data['price']

RSS_test = np.sum((model15tvs.predict(poly_15_test[features_test_15]) - poly_15_test['price'])**2)
print RSS_test













