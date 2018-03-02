import pandas as pd
import os
import numpy as np
import datetime
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
import math
from collections import defaultdict
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from CategoricalEncoderfile import CategoricalEncoder
from sklearn.preprocessing import Imputer
from scipy.sparse import coo_matrix, hstack
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, f_classif, VarianceThreshold, SelectKBest, SelectFromModel
from sklearn import linear_model
from sklearn.svm import l1_min_c
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn import metrics


pd.options.display.max_rows = 30
csvPath = "alldata.csv"

alldata = pd.read_csv(csvPath,parse_dates=['ChargeOffDate','Start_Date','End_Date','ApprovalDate'])

# Test Data
# ttt = alldata[alldata['ApprovalDate'] >= datetime.datetime(2004,1,1)]
# len(ttt.index)
# Train Data
sss = alldata[alldata['ApprovalDate'] < datetime.datetime(2004,1,1)]
len(sss.index)

# Categorical Variables
cat_attribs = ['BorrState','CDC_State','ThirdPartyLender_State',\
'subpgmdesc','ProjectState','BusinessType','MortgageCatTerm','NIACLargesBusinessSector','NIACSubsector','NIACIndustryGroup',\
'NAICSIndustries','NAICSNationalIndustries','Missing_ThirdPartyDollars']#,'BorrZip']
# Numerical Variables
num_attribs = ['GrossApproval','TermInMonths','ThirdPartyDollars',\
'MortgageAge','SPFactor','hpiFactor','unemp_rate']



# Other variables -not currently used
other_attribs = ['ChargeOffDate','MortgageID','Start_Date','End_Date']
# Our y variable
label = ['Default']

class CategoricalSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_attribs, cat_attribs):
        self.num_attribs = num_attribs
        self.cat_attribs = cat_attribs
        self.indic_prefix = []
        self.set_cat_attribs = set(cat_attribs)
        self.final_columns_set = None
        self.final_columns_list = None
        self.final_columns_index = None
        self.attrib_dict = None
        for x in cat_attribs:
            self.indic_prefix.append('Indic_' + x)
    def fit(self, df):
        s = pd.get_dummies(df[self.cat_attribs].fillna('NaN'),prefix_sep='_',prefix=self.indic_prefix,columns=self.cat_attribs,sparse=True)
        self.final_columns_index = s.columns.copy()
        self.final_columns_set = set(s.columns)
        self.final_columns_list = [x for x in s.columns]
        self.attrib_dict = dict(enumerate(self.num_attribs + self.final_columns_list))
        return self.attrib_dict
    def transform(self, df):
        s = pd.get_dummies(df[self.cat_attribs].fillna('NaN'),prefix_sep='_',prefix=self.indic_prefix,columns=self.cat_attribs,sparse=True)
        attribs_remove = set(s.columns) - self.final_columns_set
        # print('attribs_remove',attribs_remove)
        s.drop(columns=list(attribs_remove),inplace=True)
        attribs_add = self.final_columns_set - set(s.columns)
        # print('attribs_add',attribs_add)
        for y in attribs_add:
            s[y] = 0
        #re-order to be in same order as original dataframe
        return coo_matrix(s[self.final_columns_list].values)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class DataFrameSelector2(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X[self.attribute_names]
        return X.replace(np.nan,'NaNN').values

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="mean")),('std_scaler', StandardScaler())])
# cat_pipeline = Pipeline([('selector', DataFrameSelector2(cat_attribs)),('cat_encoder', CategoricalEncoder(encoding="onehot"))])
# full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline)])

data_prepared_num = num_pipeline.fit_transform(sss)
cat_encoder = CategoricalSelector(num_attribs,cat_attribs)
featureNumToName = cat_encoder.fit(sss)
catArray = cat_encoder.transform(sss)
data_prepared_numSparse = coo_matrix(data_prepared_num)
allData = hstack([data_prepared_numSparse,catArray])

y = sss[label].values
y = y.flatten()

# remove categorical variables with low variance
selector_variance = VarianceThreshold(threshold=.0025)
selector_variance.fit(allData)
c = selector_variance.get_support(indices=False)
d = selector_variance.get_support(indices=True)

featureItemize = featureNumToName.items()
featureItemize = [x for x,z in zip(featureItemize,c) if (z == 1)]
featureNumToName2 = dict([(i,x[1]) for i, x in enumerate(featureItemize)])

allDataVarThreshold = selector_variance.transform(allData)

# Perform l1 feature selection
clf_l = linear_model.LogisticRegression(C=.058, penalty='l1', tol=1e-6,max_iter=500)
std_scaler = StandardScaler()
allDataScaled = std_scaler.fit_transform(allDataVarThreshold.toarray())

clf_l.fit(allDataScaled,y)

selector_l1 = SelectFromModel(clf_l,prefit=True)
c = selector_l1.get_support(indices=False)
d = selector_l1.get_support(indices=True)

featureItemize2 = featureNumToName2.items()
featureItemize2 = [x for x,z in zip(featureItemize2,c) if (z == 1)]
featureNumToName3 = dict([(i,x[1]) for i, x in enumerate(featureItemize2)])

allDataL1 = selector_l1.transform(allDataScaled)

# Process All  Data
all_prepared_num = num_pipeline.transform(alldata)
y_all = alldata[label].values
y_all = y_all.flatten()
all_cat_data = cat_encoder.transform(alldata)
data_test_numSparse = coo_matrix(all_prepared_num)
all_data = hstack([all_prepared_num,all_cat_data])

all_data_var = selector_variance.transform(all_data)
all_data_scaled = std_scaler.transform(all_data_var.toarray())
all_data_l1 = selector_l1.transform(all_data_scaled)

# Fit model to training data. Note that the model will have better in-sample prediction than out-of-sample
clf = linear_model.LogisticRegression(C=.058, penalty='l1', tol=1e-6,max_iter=500,verbose=1)
clf.fit(allDataL1,y)

# Simulation Starts Here
# Use all_data_l1 for the data in the simulation. It has been properly formatted (mean-shifted and standardized).
# clf.predict_proba(X) will get you the probability of default
# featureNumToName3 will give you a dict for (column_number:column name), so you know what Variables
# refer to which columns of the numpy array.
