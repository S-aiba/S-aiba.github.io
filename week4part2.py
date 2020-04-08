import pandas as pd
import numpy as np

from math import sqrt

sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
train_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_train_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
test_data=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/kc_house_test_data.csv',dtype= {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})

def get_numpy_data(data, features, output):
    data['constant'] = 1
    features = ['constant'] + features
    features_matrix = data[features].as_matrix(columns=None)
    output_array = data[output].as_matrix(columns=None)
    return(features_matrix, output_array)

def predict_output(feature_matrix, weights):
    predictions=np.dot(feature_matrix,weights)
    return predictions

def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if(feature_is_constant==True):
        derivative=2*np.dot(errors,feature)
    else:
        derivative=2*np.dot(errors,feature)+2*l2_penalty*weight
    return derivative

#TESING
(example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
my_weights = np.array([1., 10.])
test_predictions = predict_output(example_features, my_weights)
errors = test_predictions - example_output

print feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False)
print np.sum(errors*example_features[:,1])*2+20.
print ''

print feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True)
print np.sum(errors)*2.

def ridge_regression_gradient_descent(feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    weights = np.array(initial_weights)
    while (max_iterations>0):
        predictions=predict_output(feature_matrix,weights)
        errors=predictions-output
        for i in xrange(len(weights)):
            if(i==0):
                feature_is_constant=True
            else:
                feature_is_constant=False
            derivative=feature_derivative_ridge(errors,feature_matrix[:,i],weights[i],l2_penalty,feature_is_constant)
            weights[i]=weights[i]-step_size*derivative
        max_iterations-=1
    return weights

simple_features = ['sqft_living']
my_output = 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, simple_features, my_output)

step_size = 1e-12
max_iterations = 1000
l2_penalty=0.0
initial_weights=np.array([0.,0.])

simple_weights_0_penalty=ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print(simple_weights_0_penalty)

l2_penalty=1e11
simple_weights_high_penalty=ridge_regression_gradient_descent(simple_feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)
print(simple_weights_high_penalty)


import matplotlib.pyplot as plt
plt.plot(simple_feature_matrix,output,'k.',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_0_penalty),'b-',
        simple_feature_matrix,predict_output(simple_feature_matrix, simple_weights_high_penalty),'r-')


initial_predictions = predict_output(simple_test_feature_matrix, initial_weights)
initial_residuals = test_output - initial_predictions
initial_RSS = (initial_residuals **2).sum()
print initial_RSS

no_regularization_predictions = predict_output(simple_test_feature_matrix, simple_weights_0_penalty)
no_regularization_residuals = test_output - no_regularization_predictions
no_regularization_RSS = (no_regularization_residuals **2).sum()
print no_regularization_RSS

regularization_predictions = predict_output(simple_test_feature_matrix, simple_weights_high_penalty)
regularization_residuals = test_output - regularization_predictions
regularization_RSS = (regularization_residuals **2).sum()
print regularization_RSS

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

step_size = 1e-12
max_iterations = 1000
initial_weights=np.array([0.,0.,0.])
l2_penalty=0.0

multiple_weights_0_penalty=ridge_regression_gradient_descent(feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)

l2_penalty=1e11
multiple_weights_high_penalty=ridge_regression_gradient_descent(feature_matrix,output,initial_weights,step_size,l2_penalty,max_iterations)

initial_predictions = predict_output(test_feature_matrix, initial_weights)
initial_residuals = test_output - initial_predictions
initial_RSS = (initial_residuals **2).sum()
print initial_RSS

no_regularization_predictions = predict_output(test_feature_matrix, multiple_weights_0_penalty)
no_regularization_residuals = test_output - no_regularization_predictions
no_regularization_RSS = (no_regularization_residuals **2).sum()
print no_regularization_RSS

regularization_predictions = predict_output(test_feature_matrix,multiple_weights_high_penalty)
regularization_residuals = test_output - regularization_predictions
regularization_RSS = (regularization_residuals **2).sum()
print regularization_RSS

print(no_regularization_predictions[0])
print (regularization_predictions[0])
print (test_output[0])
print(test_output[0]-no_regularization_predictions[0])
print (test_output[0]-regularization_predictions[0])



