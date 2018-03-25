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

# REG_PARAM = .058
# REG_PARAM = .0268
REG_PARAM = .0268
VAR_THRESH = .00001
pd.options.display.max_rows = 30
csvPath = "alldata.csv"

alldata = pd.read_csv(csvPath,parse_dates=['ChargeOffDate','Start_Date','End_Date','ApprovalDate'])

ttt = alldata[alldata['ApprovalDate'] >= datetime.datetime(2004,1,1)]
len(ttt.index)
sss = alldata[alldata['ApprovalDate'] < datetime.datetime(2004,1,1)]
len(sss.index)

# Categorical Variables
cat_attribs = ['BorrState','CDC_State','ThirdPartyLender_State',\
'subpgmdesc','ProjectState','BusinessType','MortgageCatTerm','NIACLargesBusinessSector','NIACSubsector','NIACIndustryGroup',\
'NAICSIndustries','NAICSNationalIndustries','Missing_ThirdPartyDollars','Missing_Unemp_Rate']#,'BorrZip']
# Numerical Variables
num_attribs = ['GrossApproval','TermInMonths','ThirdPartyDollars',\
'MortgageAge','SPFactor','hpiFactor','unemp_rate'] # ,'SPAnnualReturn']



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

data_prepared_num = num_pipeline.fit_transform(sss)
cat_encoder = CategoricalSelector(num_attribs,cat_attribs)
featureNumToName = cat_encoder.fit(sss)
catArray = cat_encoder.transform(sss)
data_prepared_numSparse = coo_matrix(data_prepared_num)
allData = hstack([data_prepared_numSparse,catArray])

y = sss[label].values
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

# Perform l1 feature selection
clf_l = linear_model.LogisticRegression(C=REG_PARAM, penalty='l1', tol=1e-6,max_iter=500)
# std_scaler = StandardScaler()
# allDataScaled = std_scaler.fit_transform(allDataVarThreshold.toarray())

clf_l.fit(allDataVarThreshold,y)

selector_l1 = SelectFromModel(clf_l,prefit=True)
c = selector_l1.get_support(indices=False)
d = selector_l1.get_support(indices=True)

featureItemize2 = featureNumToName2.items()
featureItemize2 = [x for x,z in zip(featureItemize2,c) if (z == 1)]
featureNumToName3 = dict([(i,x[1]) for i, x in enumerate(featureItemize2)])

allDataL1 = selector_l1.transform(allDataVarThreshold)

# Get rid of noise variables
# selector_kbest = SelectKBest(f_classif, k=140)
# selector_kbest.fit(allDataVarThreshold, y)
# c = selector_kbest.get_support(indices=False)
# d = selector_kbest.get_support(indices=True)
#
# importantVars = sorted(enumerate(selector_kbest.pvalues_),key=lambda l:l[1])
# importantVars = [x[0] for x in importantVars if x[1] < .1]
# # for x in importantVars:
# #     print(featureNumToName2[x])
#
# featureItemize2 = featureNumToName2.items()
# featureItemize2 = [x for x,z in zip(featureItemize2,c) if (z == 1)]
# featureNumToName3 = dict([(i,x[1]) for i, x in enumerate(featureItemize2)])
#
# allDataPValue = selector_kbest.transform(allDataVarThreshold)

# Analysis of quantitative data
# ys = sss['Default'].values
# fig, axes = plt.subplots(nrows=3,ncols=3)
# axReshape = axes.reshape(-1)
# sss['ThirdPartyDollars'] = sss['ThirdPartyDollars'].fillna(sss['ThirdPartyDollars'].mean())
#
# for i, quant in enumerate(num_attribs):
#     xs = sss[quant].values
#     a, b = zip( *sorted(zip(xs, ys)))
#     factor = np.array(a)
#     default = np.array(b)
#     default = np.convolve(default,np.ones(10001,)/10001.0,'valid')
#     factor = factor[5000:-5000]
#     axReshape[i].plot(factor, default, c='b')
#     axReshape[i].set_title('Default rate vs ' + str(quant))
#
# plt.subplots_adjust(hspace=.5)
# plt.savefig('num_data_summary.png')
# plt.show()

#Process Test Data
test_prepared_num = num_pipeline.transform(ttt)
y_test = ttt[label].values
y_test = y_test.flatten()
test_cat_data = cat_encoder.transform(ttt)
data_test_numSparse = coo_matrix(test_prepared_num)
all_test_data = hstack([test_prepared_num,test_cat_data])

all_test_data_var = selector_variance.transform(all_test_data)
# all_test_data_scaled = std_scaler.transform(all_test_data_var.toarray())
all_test_data_l1 = selector_l1.transform(all_test_data_var)
# all_test_data_kbest = selector_kbest.transform(all_test_data_var)

# Get ROC curve - this is without selection using L1 loss
clf = linear_model.LogisticRegression(C=REG_PARAM, penalty='l1', tol=1e-6,max_iter=500,verbose=1)
clf.fit(allData,y)

clf2 = linear_model.LogisticRegression(C=1e42, penalty='l2', tol=1e-6,max_iter=500,verbose=1)
clf2.fit(allData,y)

# predict training data
y_predict = clf.predict_proba(allData)
yy = y_predict[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y, yy, pos_label=1)
fig, ax = plt.subplots()
score_training = metrics.roc_auc_score(y, yy)
plt.plot(fpr,tpr,color='darkorange',label='Training-L1 Logistic Reg. AUC: ' + '{0:.2f}'.format(score_training))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# predict test data
y_test_predict = clf.predict_proba(all_test_data)
yy_test = y_test_predict[:,1]
fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, yy_test, pos_label=1)
score_test = metrics.roc_auc_score(y_test, yy_test)
plt.plot(fpr_test,tpr_test,color='blue',label='Test-L1 Logistic Reg. AUC: ' + '{0:.2f}'.format(score_test))

# see effects of our variable selection process

# predict training data
y_predict = clf2.predict_proba(allData)
yy = y_predict[:,1]
fpr2, tpr2, thresholds2 = metrics.roc_curve(y, yy, pos_label=1)
score_training = metrics.roc_auc_score(y, yy)
plt.plot(fpr2,tpr2,color='yellow',label='Training-Raw Data: AUC: ' + '{0:.2f}'.format(score_training))

# predict test data
y_test_predict = clf2.predict_proba(all_test_data)
yy_test = y_test_predict[:,1]
fpr_test2, tpr_test2, thresholds_test2 = metrics.roc_curve(y_test, yy_test, pos_label=1)
score_test = metrics.roc_auc_score(y_test, yy_test)
plt.plot(fpr_test2,tpr_test2,color='green',label='Test-Raw Data. AUC: ' + '{0:.2f}'.format(score_test))



legend = ax.legend(loc='best')
plt.title('ROC-Logistic Model')
plt.savefig('fooblah2.png')
plt.show()

# Get Important Variables
lossLambda = []
for i in range(allData.shape[-1]):
    print(i)
    if featureNumToName[i] not in featureNumToName3.values():
        continue
    print(featureNumToName[i])
    selector = [x for x in range(allData.shape[-1]) if x != i]
    clf_var = linear_model.LogisticRegression(C=REG_PARAM, penalty='l1', tol=1e-6,max_iter=500,verbose=1)
    all_data_l1_mod = allData.toarray()[:,selector]
    clf_var.fit(all_data_l1_mod,y)
    test_data_l1_mod = all_test_data.toarray()[:,selector]
    y_test_predict = clf_var.predict_proba(test_data_l1_mod)
    yy_test = y_test_predict[:,1]
    score_test = metrics.roc_auc_score(y_test, yy_test)
    lossLambda.append((i,featureNumToName[i],score_test))
    temp = pd.DataFrame(lossLambda)
    temp.to_csv('SelectionRank.csv',index=False,header=False)










# selection using l1_loss
# clf_l = linear_model.LogisticRegression(C=.07, penalty='l1', tol=1e-6,max_iter=500)
lossLambda = []

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(sss, sss["MortgageAge"]):
    strat_train_set = sss.iloc[train_index]
    strat_test_set = sss.iloc[test_index]
    strat_y_train = y[train_index]
    strat_y_test = y[test_index]

# Prepare train set
data_prepared_train = num_pipeline.fit_transform(strat_train_set)
cat_encoder = CategoricalSelector(num_attribs,cat_attribs)
featureNumToName = cat_encoder.fit(strat_train_set)
catArray = cat_encoder.transform(strat_train_set)
data_prepared_numSparse = coo_matrix(data_prepared_train)
allData_train = hstack([data_prepared_numSparse,catArray])

# Remove Low variance categorical variables
selector_variance = VarianceThreshold(threshold=VAR_THRESH)
selector_variance.fit(allData_train)
c = selector_variance.get_support(indices=False)
d = selector_variance.get_support(indices=True)

featureItemize = featureNumToName.items()
featureItemize = [x for x,z in zip(featureItemize,c) if (z == 1)]
featureNumToName2 = dict([(i,x[1]) for i, x in enumerate(featureItemize)])

allDataVarThreshold_train = selector_variance.transform(allData_train)
std_scaler = StandardScaler()
std_scaler.fit(allDataVarThreshold_train.toarray())
allDataScaled_train = std_scaler.transform(allDataVarThreshold_train.toarray())

# fit test data
test_prepared_num = num_pipeline.transform(strat_test_set)
test_cat_data = cat_encoder.transform(strat_test_set)
data_test_numSparse = coo_matrix(test_prepared_num)
all_test_data = hstack([test_prepared_num,test_cat_data])

all_test_data_var = selector_variance.transform(all_test_data)
all_test_data_scaled = std_scaler.transform(all_test_data_var.toarray())

lossLambda = []
cs = np.linspace(.01, .4,40)
for i in cs:
    print(i)
    clf_var = linear_model.LogisticRegression(C=i, penalty='l1', tol=1e-6,max_iter=500,verbose=1)
    clf_var.fit(allData_train,strat_y_train)
    y_test_predict = clf_var.predict_proba(all_test_data)
    yy_test = y_test_predict[:,1]
    score_test = metrics.log_loss(strat_y_test, yy_test)
    lossLambda.append((i,score_test))
    temp = pd.DataFrame(lossLambda)
    temp.to_csv('logloss.csv',index=False,header=False)


clf_l.fit(allDataScaled_train,strat_y_train)
strat_y_test_proba = clf_l.predict_proba(all_test_data_scaled)


print(lossLambda)
print('loss Lambda')
for x in lossLambda:
    print(x)
