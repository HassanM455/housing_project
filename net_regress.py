from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import pandas as pd
import numpy as np
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
from datetime import datetime
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import stats, linalg
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.model_selection import train_test_split


# function returns the header of a dataframe
def get_column_header(dframe):
    cols = np.array(dframe.columns)
    return cols


# function returns a 2D array containing unique values of a feature of a data frame
# and their unique integer labels as a list of ordered pairs
def get_unique_mapping(dframe , col_name):
    sample = dframe[col_name].unique()
    couple = list(zip(range(1,len(sample)+1),sample))
    couple = np.array(couple)
    return couple

# lines to grab data description for project
# address = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle/5407/data_description.txt?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1520314918&Signature=b%2BfhfbEV%2FPs4LKalcyGPZEViB0BMEUBMyZJJml7rMR03eRadQNpVITZVByfe9XRk02WopKHK%2Fpv7wsyw6K9Y60xG%2BXGKqQ3fAezPQverOHGOlQ8nf2b%2FLGyiChak%2BJrL3zT7wGQ9PE%2FfLUmIRjUiGYtfUItCqgo44wGXmXM648dBd5BxnbKLPt1pP5T9AzyIFYVsnYSksSLUrDLCVWBLgb2KXy71zUZKSr3C%2BWEz5zM%2BpPGg%2Fge3y47HGn0uxCse0hiOG3%2BfQe3A9eWKvl9HB4dRWlZTRIfSeMfvfcBIkowQ0mgMvL%2Bkpp8VG0lut5yLFf2UgLgQCDyElQTLVS2fFQ%3D%3D'
# temp = urlopen(address)
# site = temp.read()
# temp.close()
# soup = bs(site,'html.parser')
# data_info = soup.prettify()

# reading in csv file containing data as a Pandas data frame
file1 = pd.read_csv("C://Users/Mahmh/Desktop/housingProject/train.csv")
n_file = pd.read_csv("C://Users/Mahmh/Desktop/housingProject/train.csv")



# contains the data features
index = get_column_header(file1)


# printing statistics of features which contain NaN values
# print(n_file[[index[3], index[26], index[59]]].describe())


# replacing non-numerical feature values (i.e. string data types) with their integer labels
for count in range(len(index)):
    header = index[count]
    col_length = len(n_file[header])
    unique_set = get_unique_mapping(n_file, header)
    if type(unique_set[1][1]) == np.str_ :
        if count == 3 :
            print('skipped it!')
        for i,j in unique_set :
            for k in range(col_length):
                if str(n_file.at[k, header]) == j :
                    n_file.at[k, header] = np.float64(i)

# dropping the id's feature from data frame, so we may turn data into 2-D numpy array
# the row index of the numpy array will now be our id, or example number
n_file_ID_dropped = n_file.drop(['Id'], 1)

# beginning to create target numpy array





# dropping id's feature and LotFrontage feature since it contains over 200 Nan values
# the feature gives the total linear feet of street connected to the property
# Also eliminating the target feature, SalePrice, from data set
n_file_ID_LotFrontage_dropped = n_file.drop(['Id',index[3],'SalePrice'],1)

# first training set created by dropping Id feature and LotFrontage feature
# X_train still contains around 80 rows containing at least one NaN value
# for the features MasVnrArea and GarageYrBlt, with majority going to GarageYrBlt

X_train = np.array(n_file_ID_LotFrontage_dropped)
n_file_SalePrice = n_file['SalePrice']
Y_train = np.array(n_file_SalePrice)
where = []
col = []
# loop to figure out which rows have Nan Values in X_train
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        if np.isnan(X_train[i,j]) == True  :
            where.append([i,j,1])
            col.append(j)
        if np.isinf(X_train[i,j]) == True :
            where.append([i,j,2])
            col.append(j)
else :
    if where == [] :
        print('no nans!!')


# accumulating which rows to drop from X_train to have pure set
rows_erased_from_train = [where[i][0] for i in range(len(where))]
print(index[3], index[26], index[59])

# dropping rows from X_train containing NaN values
n_file_clean_89rowsdropped = n_file_ID_LotFrontage_dropped.drop(rows_erased_from_train)
n_file_clean_SalePrice_89rowsdropped = n_file_SalePrice.drop(rows_erased_from_train)

n_file2 = n_file_clean_89rowsdropped.join(n_file_clean_SalePrice_89rowsdropped)

index_clean = n_file_clean_89rowsdropped.columns

checking = np.array(n_file_clean_89rowsdropped)



X_train_clean_89rows = np.array(n_file_clean_89rowsdropped)
Y_train_clean_89rows = np.array(n_file_clean_SalePrice_89rowsdropped)
where1 = []
# checking to see if X_train_clean_89rows is actually pure
for i in range(len(X_train_clean_89rows)):
    for j in range(len(X_train_clean_89rows[i])):
        if np.isnan(X_train_clean_89rows[i,j]) == True  :
            where1.append([i,j,1])
        if np.isinf(X_train_clean_89rows[i,j]) == True :
            where1.append([i,j,2])


# feature scaling X_train_clean_89rows
XScaled_train_clean_89rows = preprocessing.scale(X_train_clean_89rows)

print('____________________________________________')









_xtrain , _xCVtest, _ytrain, _yCVtest = train_test_split(XScaled_train_clean_89rows,
                               Y_train_clean_89rows,
                               train_size= 0.85,
                               random_state= 0,
                               shuffle= True)
tf_train_Dframe = pd.DataFrame(_xtrain)
tf_train_Dframe.columns = index_clean

tf_target_Dframe = pd.DataFrame(_ytrain)
tf_target_Dframe.columns = ['Target']

tf_CVinput_Dframe = pd.DataFrame(_xCVtest)
tf_CVinput_Dframe.columns = index_clean

tf_CVoutput_Dframe = pd.DataFrame(_yCVtest)
tf_CVoutput_Dframe.columns = ['Target']



'''_________________________________________________________________________'''


def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
      y=pd.Series(data_set[LABEL].values),
      num_epochs=num_epochs,
      shuffle=shuffle)
num_ep = 1

test = tf.estimator.inputs.pandas_input_fn(x = tf_train_Dframe,
                                    y = pd.Series(tf_target_Dframe['Target'].values),
                                    num_epochs =1,
                                    shuffle=True)


# train
feature_cols = [tf.feature_column.numeric_column(k) for k in index_clean]

layers = [300 for i in range(4)]


regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=layers,
                                      model_dir='/data/titan/a/jaspal/Desktop/mol_dir')

regressor.train(input_fn=test, steps=100000)

print("Done training")

eval_ =  tf.estimator.inputs.pandas_input_fn(x = tf_CVinput_Dframe,
                                    y = pd.Series(tf_CVoutput_Dframe['Target'].values),
                                    num_epochs = 1,
                                    shuffle=False)

ev = regressor.evaluate(
  input_fn=eval_)
loss_score = ev["loss"]
print("Loss: {0:f}".format(loss_score) , 'epochs : %s' % num_ep )

