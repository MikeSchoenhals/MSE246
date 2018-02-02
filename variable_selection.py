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

pd.options.display.max_rows = 30
csvPath = "training.csv"

sss = pd.read_csv(csvPath,parse_dates=['ChargeOffDate','Start_Date'])

# Categorical Variables
cat_attribs = ['BorrState','BorrZip','CDC_State','CDC_Zip','ThirdPartyLender_State',\
'subpgmdesc','ProjectState','BusinessType','MortgageCatTerm','NIACLargesBusinessSector','NIACSubsector','NIACIndustryGroup',\
'NAICSIndustries','NAICSNationalIndustries','Missing_ThirdPartyDollars']
# Numerical Variables
num_attribs = ['GrossApproval','TermInMonths','TermInMonths','Adj Close','unemp_rate']
# Other variables -not currently used
other_attribs = ['ChargeOffDate','MortgageID','Start_Date']
# Our y variable
label = ['Default']

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

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', Imputer(strategy="median")),('std_scaler', StandardScaler())])
cat_pipeline = Pipeline([('selector', DataFrameSelector2(cat_attribs)),('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))])
full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline)])

data_prepared = full_pipeline.fit_transform(sss)

# Handle all of the numerical variables
