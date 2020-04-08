import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week6/kc_house_data_small.csv',dtype=dtype_dict)
train=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week6/kc_house_data_small_train.csv',dtype=dtype_dict)
test=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week6/kc_house_data_small_test.csv',dtype=dtype_dict)
validation=pd.read_csv('/home/saiba/Desktop/Machine Learning/panda files/week6/kc_house_data_validation.csv',dtype=dtype_dict)

def get_numpy_data(data, features, output):
    data['constant'] = 1
    features = ['constant'] + features
    features_matrix = data[features].as_matrix(columns=None)
    output_array = data[output].as_matrix(columns=None)
    return(features_matrix, output_array)

def normalize_features(feature_matrix):
    norms = np.sqrt(np.sum(feature_matrix**2,axis=0))
    normalized_features = feature_matrix/norms
    return (normalized_features, norms)

feature_list = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront',
                'view',
                'condition',
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built',
                'yr_renovated',
                'lat',
                'long',
                'sqft_living15',
                'sqft_lot15']
output='price'
train['floors']=train['floors'].astype(float)
test['floors']=test['floors'].astype(float)
validation['floors']=validation['floors'].astype(float)

features_train,output_train=get_numpy_data(train,feature_list,output)
features_test,output_test=get_numpy_data(test,feature_list,output)
features_validation,output_validation=get_numpy_data(validation,feature_list,output)

features_train_normalized,norms=normalize_features(features_train)
features_test_normalized=features_test/norms
features_validation_normalized=features_validation/norms

print features_test_normalized[0]
print features_train_normalized[9]

print(np.sqrt(np.sum((features_train_normalized[9]-features_test_normalized[0])**2)))

dist=[]
for i in range(0,10):
    eud=(np.sqrt(np.sum((features_train_normalized[i]-features_test_normalized[0])**2)))
    dist.append((eud,i))

print dist

dist.sort()
print dist[0]

results=features_train_normalized[0:3]-features_test_normalized[0]

print (results[0]-(features_train_normalized[0]-features_test_normalized[0]))


diff=features_train_normalized-features_test_normalized[0]
print (diff[-1].sum())

def compute_distances(features_instances, features_query):
    diff=features_instances-features_query
    distances=np.sqrt(np.sum(diff**2,axis=1))
    return distances

distance=compute_distances(features_train_normalized,features_test_normalized[2])
print min(distance)

minindex=np.argmin(distance)
print(minindex)

print output_train[382]

def k_nearest_neighbors(k, feature_train, features_query):
    distance=compute_distances(feature_train,features_query)
    return np.argsort(distance)[:k]

k_neighbours=k_nearest_neighbors(4,features_train_normalized,features_test_normalized[2])
print k_neighbours

def predict_output_of_query(k, features_train, output_train, features_query):
    k_neighbours=k_nearest_neighbors(k,features_train,features_query)
    output=np.sum(output_train[k_neighbours])/k
    return output

output_k=predict_output_of_query(4,features_train_normalized,output_train,features_test_normalized[2])
print output_k
print output_test[2]

def predict_output(k, features_train, output_train, features_query):
    row=features_query.shape[0]
    predicted_price=[]
    for i in range(row):
        predicted_price.append((predict_output_of_query(k,features_train,output_train,features_query[i]),i))
    return predicted_price

predicted_values=predict_output(10,features_train_normalized,output_train,features_test_normalized[0:10])

predicted_values.sort()
print predicted_values[0]
print output_test[6]

def predict_output_k(k, features_train, output_train, features_query):
    row=features_query.shape[0]
    predicted_price=[]
    for i in range(row):
        predicted_price.append(predict_output_of_query(k,features_train,output_train,features_query[i]))
    return predicted_price

rss_all = []
for k in range(1,16):
    predict_value = predict_output_k(k, features_train_normalized, output_train, features_validation_normalized)
    residual = (output_validation - predict_value)
    rss = sum(residual**2)
    rss_all.append(rss)

print(rss_all)

print (rss_all.index(min(rss_all)))

kvals = range(1, 16)
plt.plot(kvals, rss_all,'bo-')

predict_value=predict_output_k(7,features_train_normalized,output_train,features_test_normalized)
residual=output_test-predict_value
rss=np.sum(residual**2)
print rss