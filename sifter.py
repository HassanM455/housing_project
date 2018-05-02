from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split

def label_creator(dframe, column , labels, filters):
	labs = np.array([[0]])
	numOfLabels = len(labels)
	for i in labels:
		if i < max(labels):
			temp = dframe[(filters[i]<=dframe[column]) & (dframe[column]<filters[i+1])]
			length = len(temp)
			temp2 = np.array([i for j in range(length)])
			temp2 = np.reshape(temp2,(length,1))
			labs = np.append(temp2,labs,axis=0)
		else : 
			temp = dframe[(filters[i]<dframe[column]) & (dframe[column]<=filters[i+1])]
			length = len(temp)
			temp2 = np.array([i for j in range(length)])
			temp2 = np.reshape(temp2,(length,1))
			labs = np.append(temp2,labs,axis=0)
	else:
		labs = np.delete(labs,0,axis=0)
		labs = labs.flatten()
		labs = labs[::-1]
		return labs

def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)


def weights_to_dictionary(classifier):
	temp_dict = {}
	names = classifier.get_variable_names()
	for i in range(len(names)-3):
		lay = names[i]
		temp_dict[lay] = classifier.get_variable_value(lay)
	else:
		return temp_dict






n_file = pd.read_csv("C://Users/Mahmh/Desktop/housingProject/cleanUpData.csv")
print(n_file.describe().SalePrice)



filter_min = n_file.describe().SalePrice.values[3]
filter_limit_25 = n_file.describe().SalePrice.values[4]
filter_limit_50 = n_file.describe().SalePrice.values[5]
filter_limit_75 = n_file.describe().SalePrice.values[6]
filter_max = n_file.describe().SalePrice.values[7]




corr = n_file.corr()
correlated_columns = corr[(corr.SalePrice > 0.3)].SalePrice.index

n_file = n_file[correlated_columns.values]


n_file.sort_values('SalePrice',axis=0,ascending=True, inplace=True)
n_file.index = np.array(range(len(n_file)))





ids = [0,1,2,3]
filters = [filter_min, filter_limit_25, filter_limit_50 , filter_limit_75, filter_max]
labels = label_creator(n_file,'SalePrice',ids,filters)

n_file.drop(['SalePrice'],axis=1,inplace=True)
print(n_file.columns)

temp_train = np.array(n_file)
train = preprocessing.scale(temp_train)
idx = n_file.columns.values

_xtrain , _xCVtest, _ytrain, _yCVtest = train_test_split(train,
                               labels,
                               train_size= 0.9,
                               random_state= 0,
                               shuffle= True)


tf_train_Dframe = pd.DataFrame(_xtrain, columns=idx)
tf_target_Dframe = pd.DataFrame(_ytrain,columns=['Target'])

tf_CVinput_Dframe = pd.DataFrame(_xCVtest,columns=idx)
tf_CVoutput_Dframe = pd.DataFrame(_yCVtest,columns=['Target'])


'''_________________________________________________________________________________________________________'''



num_ep = 30

train_data = tf.estimator.inputs.pandas_input_fn(x = tf_train_Dframe,
                                    y = tf_target_Dframe['Target'],
                                    
                                    num_epochs =num_ep,
                                    shuffle=True)


# train
feature_cols = [tf.feature_column.numeric_column(k) for k in idx]

layers = [10 for i in range(6)]
print(layers)
print(layers)
print(num_ep)
print(len(tf_train_Dframe))
classifier = tf.estimator.DNNClassifier(feature_columns=feature_cols,
                                      hidden_units=layers,
                                      model_dir='C://Users/Mahmh/Desktop/housingProject/model_dir',
                                      n_classes=4)



eval_ =  tf.estimator.inputs.pandas_input_fn(x = tf_train_Dframe,
                                    y = tf_target_Dframe['Target'],
                                    num_epochs=1,
                                    shuffle=False)



eval_2 =  tf.estimator.inputs.pandas_input_fn(x = tf_CVinput_Dframe,
                                    y = tf_CVoutput_Dframe['Target'],
                                    num_epochs=1,
                                    shuffle=False)


classifier.train(input_fn=train_data, steps=5000)
pred = classifier.predict(input_fn=eval_2)
print(" \n \n \t Done training 1   C=('_'Q)  \n \n ")
weights1 = weights_to_dictionary(classifier)
val = next(pred)
print(tf_CVinput_Dframe.iloc[0,:])
print(tf_CVoutput_Dframe.iloc[0,:])
print(type(val))
print(val)

# ev1 = classifier.evaluate(input_fn=eval_, steps=100)
# names1 = classifier.get_variable_names()
# print(ev1 , '\n')


# ev4 = classifier.evaluate(input_fn=eval_, steps=10)
# print(ev4, '\n')

# for i in range(50): 
# 	classifier.train(input_fn=train_data, steps=1000)
# 	print(i)
# print(" \n \n \t Done training 50   C=('_'Q)  \n \n ")
# weights2 = weights_to_dictionary(classifier)
# ev2 = classifier.evaluate(input_fn=eval_)
# names2 = classifier.get_variable_names()
# print(ev2 , '\n')


# ev3 = classifier.evaluate(input_fn=eval_2)
# print(ev3)