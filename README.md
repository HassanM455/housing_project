# housing_project

Data cleaning : only a few attriutes had no information for houses well below the median price, partly due to the fact that cheaper houses did not have a pool, mason veneer walls, etc... If the NaN values were greater than 200, I dropped the attribute. For other's, I deleted the example for the training data if the NaN count was below 100.

K_mean.py -- running k-means clustering algorithm to create clusters of data points, then attempting to do statistical analysis on each cluster to see if we can find any hidden information. 

Sifter.py -- contains a classification based neural net that places houses in 4 categories, which are based of the quartiles of the training data's 'SalePrice' attribute. Currently, at best, it has achieved an accuracy score of 90% based on a specific topology of the architecture. Still in the process of hypertuning the parameters 

anamoly.py -- used to detect anomalies in the traning data using the Random Isolation Forests algorithm using the Scikit-Learn library. Going to see if training on data without anomalies can potentially decrease the variance of the net, therefore hopefully lowering the error on the CV testing data for the regression net. 

net_regress.py -- contains regression neural net. Trying different topologies of different architectures. 

correlated_net.py -- contains regression net using attributes that have a correlation value above a certain threshold, with the 'SalePrice' attribute. 

Objective :  Currently, I am exploring creating two different nets. The first net would classify whether a house is going to sell above a  certain price or below (targeting $375,000). Each output of this net would lead to a set of regression based neural nets. For houses classified above the selected price, I will be able to bring back the attributes I had dropped due to NaN values in the cheaper house examples, since for houses above $375,000, the attribute showed the highest correlation value with the 'SalePrice' of the house (around 0.78). 

The above experiment is a trial to get used to implementing neural nets. I'm currently in the process of scrapping data from real-estate websites and various other sources to train my net on. The trial net is being used as a starting point for this projects neural network. I will have new attributes for this data such as : Average household income of the neighborhood, school district rating, school distance, distance from public transportation, etc... 
