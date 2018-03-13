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
from sklearn.linear_model import LinearRegression
from sklearn.svm import l1_min_c
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn import metrics

# REG_PARAM = .058
# REG_PARAM = .0268
REG_PARAM = .0268
VAR_THRESH = .00001
pd.options.display.max_rows = 30
csvPath = "alldata.csv"

alldata = pd.read_csv(csvPath,parse_dates=['ChargeOffDate','Start_Date','End_Date','ApprovalDate'])

alldata['LossPercentage'] = alldata['GrossChargeOffAmount']/alldata['GrossApproval']

# Categorical Variables
cat_attribs = ['BorrState','CDC_State','ThirdPartyLender_State',\
'subpgmdesc','ProjectState','BusinessType','MortgageCatTerm','NIACLargesBusinessSector','NIACSubsector','NIACIndustryGroup',\
'NAICSIndustries','NAICSNationalIndustries','Missing_ThirdPartyDollars','Missing_Unemp_Rate']#,'BorrZip']
# Numerical Variables
num_attribs = ['GrossApproval','TermInMonths','ThirdPartyDollars',\
'MortgageAge','SPFactor','hpiFactor','unemp_rate'] # ,'SPAnnualReturn']

# Other variables -not currently used
other_attribs = ['GrossChargeOffAmount', 'ChargeOffDate','MortgageID','Start_Date','End_Date', 'ApprovalDate']

# Our y variable
label = ['LossPercentage']

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
        s = pd.get_dummies(df[self.cat_attribs].fillna('NaN'),prefix_sep='_',drop_first=True, prefix=self.indic_prefix,columns=self.cat_attribs,sparse=True)
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

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="mean")),('std_scaler', StandardScaler())])
# cat_pipeline = Pipeline([('selector', DataFrameSelector2(cat_attribs)),('cat_encoder', CategoricalEncoder(encoding="onehot"))])
# full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline)])

data_prepared_num = num_pipeline.fit_transform(alldata)
cat_encoder = CategoricalSelector(num_attribs,cat_attribs)
featureNumToName = cat_encoder.fit(alldata)
catArray = cat_encoder.transform(alldata)
data_prepared_numSparse = coo_matrix(data_prepared_num)
allData = hstack([data_prepared_numSparse,catArray])
y = alldata[label].values
y = y.flatten()


# remove categorical variables with low variance
selector_variance = VarianceThreshold(threshold=VAR_THRESH)
# selector_variance = VarianceThreshold(threshold=.0025)
selector_variance.fit(allData)
c = selector_variance.get_support(indices=False)
d = selector_variance.get_support(indices=True)


featureItemize = featureNumToName.items()
featureItemize = [x for x,z in zip(featureItemize,c) if (z == 1)]
featureNumToName2 = dict([(i,x[1]) for i, x in enumerate(featureItemize)])
allDataVarThreshold = selector_variance.transform(allData)

lm = LinearRegression()

lm.fit(allDataVarThreshold, y)

# R^2 for the linear model
lm.score(allDataVarThreshold, y)