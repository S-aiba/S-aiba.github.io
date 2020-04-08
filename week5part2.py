import pandas as pd
import numpy as np

import  matplotlib.pyplot as plt
from math import sqrt

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_data.csv',dtype= dtype_dict )
train_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_train_data.csv', dtype=dtype_dict)
test_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_test_data.csv',dtype=dtype_dict)
valid_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week3/wk3_kc_house_valid_data.csv',dtype=dtype_dict)

def get_numpy_data(data, features, output):
    data['constant'] = 1
    features = ['constant'] + features
    features_matrix = data[features].as_matrix(columns=None)
    output_array = data[output].as_matrix(columns=None)
    return(features_matrix, output_array)

def predict_output(feature_matrix, weights):
    predictions=np.dot(feature_matrix,weights)
    return predictions


def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix, axis=0)
    normalized_features = feature_matrix/norms
    return (normalized_features, norms)

features=['sqft_living','bedrooms']
output='price'

features_matrix,output_array=get_numpy_data(train_data,features,output)
simple_features_matrix,norms=normalize_features(features_matrix)

weights=[1,4,1]
prediction=predict_output(simple_features_matrix,weights)
ro_1 = np.dot(simple_features_matrix[:,1],(output_array - prediction + weights[1]*simple_features_matrix[:,1]))
ro_2 = np.dot(simple_features_matrix[:,2],(output_array - prediction + weights[2]*simple_features_matrix[:,2]))
print ro_1,ro_2

l1_penalty_range=[2*ro_1,2*ro_2]
print l1_penalty_range


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    prediction = predict_output(feature_matrix, weights)
    ro_i = np.dot(feature_matrix[:, i], (output - prediction + weights[i] * feature_matrix[:, i]))

    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + l1_penalty / 2
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - l1_penalty / 2
    else:
        new_weight_i = 0.

    return new_weight_i

import math
print lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],
                   [2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1)

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    max_change = tolerance*2
    weights = initial_weights
    while max_change > tolerance:
        max_change = 0
        for i in range(len(weights)):
            old_weight_i = weights[i]
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
            change = np.abs(weights[i] - old_weight_i)
            if change>max_change:
                max_change = change
    return weights


simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

simple_features_matrix,output_array=get_numpy_data(sales,simple_features,my_output)
(normalized_simple_feature_matrix, simple_norms)=normalize_features(simple_features_matrix)

(simple_feature_matrix, output) = get_numpy_data(sales, simple_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix)

new_weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)


print new_weights

print np.sum((output-predict_output(normalized_simple_feature_matrix,new_weights))**2)

train_data['floors']=train_data['floors'].astype(float)
train_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated']
output='price'
(train_feature_matrix, output_array)=get_numpy_data(train_data,train_features,output)
(train_normalized_features_matrix, normalization) =normalize_features(train_feature_matrix)

l1_penalty = 1e7
initialize_weights = np.zeros(14)
tolerance = 1

weight1e7=lasso_cyclical_coordinate_descent(train_normalized_features_matrix,output_array,initialize_weights,l1_penalty,tolerance)
print (pd.Series(weight1e7,index=['intercept']+train_features))

l1_penalty = 1e8
initialize_weights = np.zeros(14)
tolerance = 1.0

weights1e8 = lasso_cyclical_coordinate_descent(train_normalized_features_matrix, output_array,initialize_weights, l1_penalty, tolerance)
print (pd.Series(weights1e8,index=['intercept']+train_features))

l1_penalty = 1e4
tolerance = 5e5
weights1e4 = lasso_cyclical_coordinate_descent(train_normalized_features_matrix, output_array,initialize_weights, l1_penalty, tolerance)

print (pd.Series(weights1e4, index=['intercept']+train_features))

weights1e7_normalized = weight1e7 / normalization
weights1e8_normalized = weights1e8 / normalization
weights1e4_normalized = weights1e4 / normalization


test_data['floors']=test_data['floors'].astype(float)
(test_feature_matrix, test_output)=get_numpy_data(test_data,train_features,output)

print (np.sum((test_output-predict_output(test_feature_matrix,weights1e7_normalized))**2))
prediction1=predict_output(test_feature_matrix,weights1e7_normalized)
print prediction1[0]
print test_output[0]

print (np.sum((test_output-predict_output(test_feature_matrix,weights1e8_normalized))**2))

prediction1=predict_output(test_feature_matrix,weights1e8_normalized)
print prediction1[0]

print (np.sum((test_output-predict_output(test_feature_matrix,weights1e4_normalized))**2))

prediction1=predict_output(test_feature_matrix,weights1e4_normalized)
print prediction1[0]





