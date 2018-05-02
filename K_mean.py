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



'''___________________k-means by price________________________'''

def _1Darray_to_2Darray(arr):
    temp = []
    for i in range(len(arr)):
        temp.append([arr[i]])
    else :
        return np.array(temp)


def k_mean_centers_labels(n, nparray):
    k_means = KMeans(n_clusters = n,
                     init = np.array([[100000],[300000],[550000]]),
                     random_state = 0,
                     max_iter=30000 )
    k_means.fit(nparray)
    return k_means.cluster_centers_ , k_means.labels_


def label_frame_ret(arr):
    return pd.DataFrame({'labels' : arr})



#def k_mean_centers_labels2(n, nparray):
#    k_means = KMeans(n_clusters = n)
#    k_means.fit(nparray)
#    temp = k_means.cluster_centers_
#    temps = []
#    for i in temp :
#        temps.append(round(float(i),0))
#    else :
#        temps.sort()
#    return  temps , k_means.labels_


def label_frame_unique_ret(arr):
    temp = pd.DataFrame({'labels' : arr})
    return temp['labels'] , temp['labels'].unique()

def labeled_frame_generator(frame, *labs  ):
    holder = []
    for i in labs:
        holder.append(frame[(frame['labels'] == i)])
    else :
        return holder

def cluster_frame_corr(df):
    temp_corr = df.corr()
    temp_corr.sort_values('SalePrice',ascending = False, inplace = True)
    temp_corr.drop(['labels'],1 , inplace = True)
    temp_corr.drop('SalePrice', inplace = True)
    return temp_corr



Y = _1Darray_to_2Darray(Y_train_clean_89rows)

XY = np.append(X_train_clean_89rows,Y,axis = 1)
index_clean = np.append(index_clean,'SalePrice')

df = pd.DataFrame(XY)
df.columns = index_clean
newSorted = df.sort_values('SalePrice')

kY = _1Darray_to_2Darray(np.array(newSorted.SalePrice))
kY = kY.astype(np.float64)

cents , exLabs = k_mean_centers_labels(3, kY)
print('Centers are : ', cents)
exLabs = _1Darray_to_2Darray(exLabs)
XYL = np.append(XY,exLabs, axis = 1)
XYL = XYL.astype(np.float64)

index_clean = np.append(index_clean,'Label')
df_new = pd.DataFrame(XYL)
df_new.columns = index_clean

uni_labs = df_new['Label'].unique()
uni_labs = uni_labs.astype(np.int)


grouped_frames = []
for val in uni_labs :
    df1 = df_new[(df_new['Label'] == uni_labs[val])]
    grouped_frames.append(df1)



'''________________Analysis for 1'st cluster__________________'''

df1 = df1.drop('Label',1)
pca_OBJ = PCA(n_components = 3, random_state = 0)
temp = np.array(df1)
pca_OBJ.fit(temp)
print([round(i,3) for i in pca_OBJ.explained_variance_ratio_])