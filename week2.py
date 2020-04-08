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

def predict_outcome(feature_matrix, weights):
    prediction=np.dot(feature_matrix,weights)
    return prediction

def feature_derivative(errors, feature):
    derivative=2*np.dot(errors,feature)
    return derivative

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    weights = np.array(initial_weights)
    while not converged:
        predicitons=predict_outcome(feature_matrix,weights)
        error=predicitons-output
        gradient_sum_squares = 0
        for i in range(len(weights)):
            derivative=np.dot(error,feature_matrix[:,i])
            gradient_sum_squares+=(derivative**2)
            weights[i]=weights[i]-step_size*derivative
        gradient_magnitude=sqrt(gradient_sum_squares)
        if(gradient_magnitude<tolerance):
            converged=True
    return weights



simple_features = ['sqft_living']
my_output= 'price'
(simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
initial_weights = np.array([-47000., 1.])
step_size = 7e-12
tolerance = 2.5e7

simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)

print (simple_weights)

(simple_feature_matrix_test, output_test) = get_numpy_data(test_data, simple_features, my_output)

weights_test=regression_gradient_descent(simple_feature_matrix_test,output_test,initial_weights,step_size,tolerance)

predicted_price=predict_outcome(simple_feature_matrix_test,simple_weights)
print (predicted_price[0])

rss1=np.sum((predicted_price-output_test)**2)
print (rss1)

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features,my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9

(feature_matrix_test,output_test_new)=get_numpy_data(test_data,model_features,my_output)
weights_new=regression_gradient_descent(feature_matrix,output,initial_weights,step_size,tolerance)
predicted_new_price=predict_outcome(feature_matrix_test,weights_new)
print (predicted_new_price[0])

print (test_data['price'][0])


rss2=np.sum((predicted_new_price-output_test_new)**2)
print(rss2)
print (rss1>rss2)


