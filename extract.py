import pandas as pd
import os
import numpy as np
import datetime
from datetime import timedelta
from pandas.tseries.offsets import DateOffset
from dateutil.relativedelta import relativedelta
import math
from collections import defaultdict
pd.options.display.max_rows = 30

END_DATE = datetime.datetime(2014,1,1)

csvPath = "SBA_Loan_data_.csv"
s = pd.read_csv(csvPath,dtype={'ThirdPartyLender_Name': str,'ThirdPartyLender_City':str,\
'ThirdPartyLender_State':'category','BorrState':'category','BorrZip':'category','CDC_State':'category', \
'CDC_Zip':'category','subpgmdesc':'category','NaicsCode':str,'ProjectState':'category','BusinessType':'category', \
'LoanStatus':'category','ChargeOffDate':str,'ApprovalDate':str},parse_dates=['ApprovalDate','ChargeOffDate'])


# Filter loans that are cancelled, missing, or exempt. Also dropping laons that are approved after 2014 per Piazza post.
s = s[((s.LoanStatus == 'PIF') | (s.LoanStatus == 'CHGOFF')) & (s.ApprovalDate < datetime.datetime(2014,1,1))]

# Get number of years loan has survived - rounded to closest year.
# SECONDS_PER_YEAR = 31536000.0
#
# def lengthOfLoan(row):
#     if row.LoanStatus == 'CHGOFF':
#         return int(math.ceil((row['ChargeOffDate']- row['ApprovalDate']).total_seconds()/SECONDS_PER_YEAR))
#     else:
#         return int(round((END_DATE - row['ApprovalDate']).total_seconds()/SECONDS_PER_YEAR))

def lengthOfLoan2(row):
    if row.LoanStatus == 'CHGOFF':
        return row['ChargeOffDate']- row['ApprovalDate']
    else:
        return END_DATE - row['ApprovalDate']

s['CurrentLength'] =  s.apply(lengthOfLoan2,axis=1)


# Filter records that have 0 years exposure
# s = s[s.CurrentLength > 0]

# Create categorical Variable for Mortgage term length
MLengthlables = {0:'12Months or Less',1:'12To24Months',2:'24MonthsTo5Years',3:'5YearsTo10Years',4:'10YearsTo20Years',5:'MoreThan20Years'}
s['MortgageCatTerm'] = pd.cut(s['TermInMonths'], [0,12,24,60,120,240,500],labels=MLengthlables.keys())

# Parse NIAC code into categories.
def getNAICCode12(row):
    if type(row['NaicsCode']) == str:
        return str(row['NaicsCode'])[0:2]
    else:
        return np.nan

def getNAICCode3(row):
    if type(row['NaicsCode']) == str:
        return str(row['NaicsCode'])[2]
    else:
        return np.nan

def getNAICCode4(row):
    if type(row['NaicsCode']) == str:
        return str(row['NaicsCode'])[3]
    else:
        return np.nan

def getNAICCode5(row):
    if type(row['NaicsCode']) == str:
        return str(row['NaicsCode'])[4]
    else:
        return np.nan

def getNAICCode6(row):
    if type(row['NaicsCode']) == str:
        return str(row['NaicsCode'])[5]
    else:
        return np.nan

s['NIACLargesBusinessSector'] = s.apply(getNAICCode12,axis=1).astype('category')
s['NIACSubsector'] = s.apply(getNAICCode3,axis=1).astype('category')
s['NIACIndustryGroup'] = s.apply(getNAICCode4,axis=1).astype('category')
s['NAICSIndustries'] = s.apply(getNAICCode5,axis=1).astype('category')
s['NAICSNationalIndustries'] = s.apply(getNAICCode6,axis=1).astype('category')


# Add indicators for categorical variables. I currently commented this out. It's too much data to handle.
# s = pd.get_dummies(s,prefix=['Indic_NIACLargesBusinessSector','Indic_NIACSubsector',\
# 'Indic_NIACIndustryGroup','Indic_NAICSIndustries','Indic_NAICSNationalIndustries',\
# 'Indic_BorrState','Indic_ThirdPartyLender_State','Indic_BorrZip',\
# 'Indic_CDC_State','Indic_CDC_Zip','Indic_ProjectState','Indic_BusinessType','Indic_subpgmdesc'],\
# prefix_sep='_',dummy_na=True,drop_first=False,columns= \
# ['NIACLargesBusinessSector','NIACSubsector','NIACIndustryGroup','NAICSIndustries','NAICSNationalIndustries',\
# 'BorrState','ThirdPartyLender_State','BorrZip','CDC_State','CDC_Zip',\
# 'ProjectState','BusinessType','subpgmdesc'])

# Add missing indicators for quantitative variables
def mapMissingValues(row,col_label):
    if not np.isnan(row[col_label]):
        return 0
    return 1

s["Missing_ThirdPartyDollars"] = s.apply(lambda row: mapMissingValues(row,'ThirdPartyDollars'),axis=1)



# Get S&P 500 data - Loaded from spreadsheet. Data comes from yahoo finance.
start_date = '1989-01-02'
end_date = '2014-01-01'
SP500 = pd.read_csv('GSPC.csv',index_col='Date',parse_dates=['Date'])
adj_close = SP500['Adj Close']
every_day = pd.date_range(start=start_date, end=end_date, freq='D')
adj_close = adj_close.reindex(every_day)
adj_close = adj_close.fillna(method='ffill')
adj_close=adj_close.to_frame()
adj_close_return = adj_close.pct_change(freq=DateOffset(months=12))
adj_close = pd.merge(adj_close,adj_close_return,left_index=True,right_index=True,how='left')
adj_close = adj_close.rename(columns={'Adj Close_x':'SPIndex','Adj Close_y':'SPAnnualReturn'})
adj_close_rolling = adj_close.rolling(window=100).mean()
adj_close = adj_close.reset_index()
adj_close["year"] = adj_close.apply(lambda row: row['index'].year,axis=1)
adj_close["month"] = adj_close.apply(lambda row: row['index'].month,axis=1)

# Get Unemployment Rates
unemp = pd.read_csv('UnemploymentRates.csv',sep=r"\s+",header=None,index_col=0)
unemp = unemp.unstack().swaplevel()
unemp.index = unemp.index.rename(['year','month'])
unemp = unemp.reset_index()
unemp.columns = pd.Index(['year', 'month', 'unemp_rate'], dtype='object')

# read in inflation Rates - NOT CURRENTLY USED
# infl = pd.read_csv('Inflation.csv',sep=r",",header=None,index_col=0)
# infl = infl.unstack().swaplevel()
# infl.index = infl.index.rename(['year','month'])
# infl = infl.sort_index(level=0,sort_remaining=False)
# infl = infl.reset_index()
# infl.columns = pd.Index(['year', 'month', 'infl_rate'], dtype='object')
# infl['infl_factor'] = 0
# for i,x in infl.iterrows():
#     if i == 0:
#         infl['infl_factor'].iloc[i] = 1.0
#     else:
#         infl['infl_factor'].iloc[i] = infl['infl_factor'].iloc[i-1] *  ((1+infl['infl_rate'].iloc[i]/100) ** (1.0/12.0))



state_HPIndex = pd.read_csv('HPI_AT_state.csv',sep=r",",names=['State','Year','Quarter','hpi'])
state_HPIndex['Month'] = state_HPIndex.apply(lambda l: l['Quarter']*3,axis=1)
state_HPIndex['Date'] = state_HPIndex.apply(lambda l: datetime.date(int(l['Year']),int(l['Month']),1),axis=1)
end_date = '2014-01-01'
start_date = '1990-01-01'
state_HPIndex = state_HPIndex.set_index(['State','Date'])
every_month = pd.date_range(start=start_date, end=end_date, freq='D')
multi_index = pd.MultiIndex.from_product([state_HPIndex.index.levels[0],every_month])
state_HPIndex = state_HPIndex.reindex(multi_index)
state_HPIndex = state_HPIndex.drop(columns=['Year','Quarter','Month'])
state_HPIndex.index = state_HPIndex.index.rename(['State','Date'])
state_HPIndex['hpi'] = pd.to_numeric(state_HPIndex['hpi'], errors='coerce')
state_HPIndex = state_HPIndex.unstack(level=0)
state_HPIndex = state_HPIndex.interpolate(method='spline',order=3)
state_HPIndex = state_HPIndex.fillna(method='backfill')
state_HPIndex = state_HPIndex.stack(dropna=False)
state_HPIndex = state_HPIndex.unstack(level=0)
state_HPIndex = state_HPIndex.fillna(state_HPIndex.mean())
state_HPIndex = state_HPIndex.stack(dropna=False)
state_HPIndex = state_HPIndex.reset_index(level=['Date','State'])
# state_HPIndex.to_csv("hpitest.csv",index=False)


# combine all TS into one dataframe
combine = pd.merge(adj_close,unemp,left_on=['year','month'],right_on=['year','month'],how='left')

# join state_HPrice to combine
combine = pd.merge(state_HPIndex,combine,left_on=['Date'],right_on=['index'],how='left')
combine = combine.drop(columns=['year','month','index'])

#creatingMortageID - Useful for cox model when we may need to recombine
s.index.rename('MortgageID',inplace=True)
s.reset_index(inplace=True)

# Drop useless columns before expanding records - save on RAM as much as possible
s = s.drop(columns=['NaicsCode','Program','BorrName','BorrStreet','BorrCity','CDC_Name',\
'CDC_Street','CDC_City','ThirdPartyLender_Name','ThirdPartyLender_City','InitialInterestRate','NaicsDescription','DeliveryMethod',\
'ProjectCounty'])
# Create multiple records from single record and join time dependent data
framesToStack = []
count = 0
def defaultMap(row):
    if row.LoanStatus == 'CHGOFF':
        if row['ChargeOffDate'] < (row['Start_Date'] + relativedelta(years=1)):
            return 1
    return 0

while(True):
    s['Start_Date'] = s.apply(lambda row: row['ApprovalDate'] + relativedelta(years=count),axis=1)
    p = s[s.CurrentLength+s.ApprovalDate > s.Start_Date]
    p['MortgageAge'] = count
    if len(p.index) == 0:
        break
    p['End_Date'] = p.apply(lambda row: min(row['ApprovalDate'] + relativedelta(years=count+1),row['ApprovalDate'] + row['CurrentLength']),axis=1)
    p = pd.merge(p,combine,left_on=['BorrState','Start_Date'],right_on=['State','Date'],how='left')
    p = pd.merge(p,combine,left_on=['BorrState','ApprovalDate'],right_on=['State','Date'],how='left')
    print('got here')
    p['Default'] = p.apply(defaultMap,axis=1)
    framesToStack.append(p)
    count = count+1

t = pd.concat(framesToStack,axis=0)

# Normalize S&P Index and HPI
t['SPFactor'] = t.apply(lambda l: l['SPIndex_x']/l['SPIndex_y'],axis=1)
t['SPAnnualReturn'] = t['SPAnnualReturn_x']
t['hpiFactor'] = t.apply(lambda l: l['hpi_x']/l['hpi_y'],axis=1)
t['unemp_rate'] = t['unemp_rate_x']



# Drop useless Data - don't currently see the need for it.
t = t.drop(columns=['CurrentLength','ApprovalDate','ApprovalFiscalYear',\
'GrossChargeOffAmount','LoanStatus','SPIndex_x','SPIndex_y','SPAnnualReturn_x',\
'SPAnnualReturn_y','hpi_x','hpi_y','State_x','State_y','Date_x','Date_y','unemp_rate_x','unemp_rate_y'])


 # Write output types to csv

# Create Test Set and Training/Validation Set
testSet = t[t['Start_Date'] >= datetime.datetime(2010,1,1)]
trainingSet = t[t['Start_Date'] < datetime.datetime(2010,1,1)]
print(len(testSet.index)/len(t.index))
testSet.to_csv("test.csv",index=False)
trainingSet.to_csv("training.csv",index=False)
