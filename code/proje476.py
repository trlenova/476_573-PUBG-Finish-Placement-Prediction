import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import os

train_data = pd.read_csv('../bil476proje/input/train_V2.csv',nrows=40000)
test = pd.read_csv('../bil476proje/input/test_V2.csv',nrows=10000)


train_data.drop(train_data[train_data['winPlacePerc']==np.NAN].index,inplace = True)

train_data['matchType'] = train_data['matchType'].map({
    'crashfpp':0,
    'crashtpp':0,
    'duo':1,
    'duo-fpp':1,
    'flarefpp':4,
    'flaretpp':4,
    'normal-duo':1,
    'normal-duo-fpp':1,
    'normal-solo':2,
    'normal-solo-fpp':2,
    'normal-squad':3,
    'normal-squad-fpp':3,
    'solo':2,
    'solo-fpp':2,
    'squad':3,
    'squad-fpp':3
    })


train_data['playerJoined'] = train_data.groupby('matchId')['matchId'].transform('count')
train_data['killsNorm'] = train_data['kills']*((train_data['playerJoined'] - 1)/(100-1))
train_data['damageDealtNorm'] = train_data['damageDealt'] * ((train_data['playerJoined'] - 1)/99)

#Rank
match = train_data.groupby('matchId')
train_data['killsPerc'] = match['kills'].rank(pct=True).values
train_data['killPlacePerc'] = match['killPlace'].rank(pct=True).values
train_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
train_data['walkPerc_killsPerc'] = train_data['walkDistancePerc'] / train_data['killsPerc']

#distance
train_data['totalDistance'] = train_data['rideDistance'] + train_data['swimDistance'] + train_data['walkDistance']


#
train_data['_healthItems'] = train_data['heals'] + train_data['boosts']
train_data['_headshotKillRate'] = train_data['headshotKills'] / train_data['kills']
train_data['_killPlaceOverMaxPlace'] = train_data['killPlace'] / train_data['maxPlace']
train_data['_killsOverWalkDistance'] = train_data['kills'] / train_data['walkDistance']

train_data[train_data == np.Inf] = np.NaN
train_data[train_data == np.NINF] = np.NaN
train_data.fillna(0, inplace=True)


#cheaters
train_data['killWithoutMove'] = ((train_data['kills'] > 0) & (train_data['totalDistance'] == 0))
train_data.drop(train_data[train_data['killWithoutMove'] == True].index,inplace = True)



train_data.drop(train_data[train_data['boosts'] >11].index,inplace = True)
train_data.drop(train_data[train_data['weaponsAcquired'] >20].index,inplace = True)

train_data.loc[(train_data['rankPoints']==-1), 'rankPoints'] = 0
train_data['_killPoints_rankpoints'] = train_data['rankPoints']+train_data['killPoints']

columns=list(train_data.columns)
columns.remove("Id")
columns.remove("matchId")
columns.remove("groupId")
columns.remove("matchType")
columns.remove("winPlacePerc")
columns.remove("killWithoutMove")

##mean columns
meanData = train_data.groupby(['matchId','groupId'])[columns].agg('mean')
meanData = meanData.replace([np.inf, np.NINF,np.nan], 0)
meanDataRank = meanData.groupby('matchId')[columns].rank(pct=True).reset_index()
train_data = pd.merge(train_data, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del meanData
train_data = train_data.drop(["vehicleDestroys_mean","rideDistance_mean","roadKills_mean","rankPoints_mean"], axis=1)
train_data = pd.merge(train_data, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId'])
del meanDataRank
train_data = train_data.drop(["numGroups_meanRank","rankPoints_meanRank"], axis=1)
train_data = train_data.join(train_data.groupby('matchId')[columns].rank(ascending=False).add_suffix('_rankPlace').astype(int))

##max
maxData = train_data.groupby(['matchId','groupId'])[columns].agg('max')
maxDataRank = maxData.groupby('matchId')[columns].rank(pct=True).reset_index()
train_data = pd.merge(train_data, maxData.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del maxData
train_data = train_data.drop(["assists_max","killPoints_max","headshotKills_max","numGroups_max","revives_max","teamKills_max","roadKills_max","vehicleDestroys_max"], axis=1)
train_data = pd.merge(train_data, maxDataRank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
del maxDataRank
train_data = train_data.drop(["roadKills_maxRank","matchDuration_maxRank","maxPlace_maxRank","numGroups_maxRank"], axis=1)

##min
minData = train_data.groupby(['matchId','groupId'])[columns].agg('min')
minDataRank = minData.groupby('matchId')[columns].rank(pct=True).reset_index()
train_data = pd.merge(train_data, minData.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del minData
train_data = train_data.drop(["heals_min","killStreaks_min","killPoints_min","maxPlace_min","revives_min","headshotKills_min","weaponsAcquired_min","rankPoints_min","matchDuration_min","teamKills_min","numGroups_min","assists_min","roadKills_min","vehicleDestroys_min"], axis=1)
train_data = pd.merge(train_data, minDataRank, suffixes=["", "_minRank"], how='left', on=['matchId', 'groupId'])
del minDataRank
train_data = train_data.drop(["killPoints_minRank","matchDuration_minRank","maxPlace_minRank","numGroups_minRank"], axis=1)


#number of grubs
groupSize = train_data.groupby(['matchId','groupId']).size().reset_index(name='group_size')
train_data = pd.merge(train_data, groupSize, how='left', on=['matchId', 'groupId'])
del groupSize






train = train_data.drop(['Id','groupId','matchId','killWithoutMove','kills','damageDealt','numGroups','swimDistance','playerJoined'],axis = 1)

Y_train = train['winPlacePerc']
X_train = train.drop(['winPlacePerc'],axis = 1)


Y_train.head()

from sklearn.metrics import mean_absolute_error

m1 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features=0.5,n_jobs=-1)
m1.fit(X_train, Y_train)
mean_absolute_error(m1.predict(X_train), Y_train)


X_test = test.copy()

X_test['matchType'] = X_test['matchType'].map({
    'crashfpp':0,
    'crashtpp':0,
    'duo':1,
    'duo-fpp':1,
    'flarefpp':4,
    'flaretpp':4,
    'normal-duo':1,
    'normal-duo-fpp':1,
    'normal-solo':2,
    'normal-solo-fpp':2,
    'normal-squad':3,
    'normal-squad-fpp':3,
    'solo':2,
    'solo-fpp':2,
    'squad':3,
    'squad-fpp':3
    })

X_test['playerJoined'] = X_test.groupby('matchId')['matchId'].transform('count')
X_test['killsNorm'] = X_test['kills']*((X_test['playerJoined'] - 1)/(100-1))
X_test['damageDealtNorm'] = X_test['damageDealt'] * ((X_test['playerJoined'] - 1)/99)

#Rank
match = X_test.groupby('matchId')
X_test['killsPerc'] = match['kills'].rank(pct=True).values
X_test['killPlacePerc'] = match['killPlace'].rank(pct=True).values
X_test['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values
X_test['walkPerc_killsPerc'] = X_test['walkDistancePerc'] / X_test['killsPerc']

#distance
X_test['totalDistance'] = X_test['rideDistance'] + X_test['swimDistance'] + X_test['walkDistance']


#
X_test['_healthItems'] = X_test['heals'] + X_test['boosts']
X_test['_headshotKillRate'] = X_test['headshotKills'] / X_test['kills']
X_test['_killPlaceOverMaxPlace'] = X_test['killPlace'] / X_test['maxPlace']
X_test['_killsOverWalkDistance'] = X_test['kills'] / X_test['walkDistance']



X_test.loc[(X_test['rankPoints']==-1), 'rankPoints'] = 0
X_test['_killPoints_rankpoints'] = X_test['rankPoints']+X_test['killPoints']

X_test[X_test == np.Inf] = np.NaN
X_test[X_test == np.NINF] = np.NaN
X_test.fillna(0, inplace=True)

columns=list(X_test.columns)
columns.remove("Id")
columns.remove("matchId")
columns.remove("groupId")
columns.remove("matchType")

##mean columns
meanData = X_test.groupby(['matchId','groupId'])[columns].agg('mean')
meanData = meanData.replace([np.inf, np.NINF,np.nan], 0)
meanDataRank = meanData.groupby('matchId')[columns].rank(pct=True).reset_index()
X_test = pd.merge(X_test, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])
del meanData
X_test = X_test.drop(["vehicleDestroys_mean","rideDistance_mean","roadKills_mean","rankPoints_mean"], axis=1)
X_test = pd.merge(X_test, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId'])
del meanDataRank
X_test = X_test.drop(["numGroups_meanRank","rankPoints_meanRank"], axis=1)
X_test = X_test.join(X_test.groupby('matchId')[columns].rank(ascending=False).add_suffix('_rankPlace').astype(int))

##max
maxData = X_test.groupby(['matchId','groupId'])[columns].agg('max')
maxDataRank = maxData.groupby('matchId')[columns].rank(pct=True).reset_index()
X_test = pd.merge(X_test, maxData.reset_index(), suffixes=["", "_max"], how='left', on=['matchId', 'groupId'])
del maxData
X_test = X_test.drop(["assists_max","killPoints_max","headshotKills_max","numGroups_max","revives_max","teamKills_max","roadKills_max","vehicleDestroys_max"], axis=1)
X_test = pd.merge(X_test, maxDataRank, suffixes=["", "_maxRank"], how='left', on=['matchId', 'groupId'])
del maxDataRank
X_test = X_test.drop(["roadKills_maxRank","matchDuration_maxRank","maxPlace_maxRank","numGroups_maxRank"], axis=1)

##min
minData = X_test.groupby(['matchId','groupId'])[columns].agg('min')
minDataRank = minData.groupby('matchId')[columns].rank(pct=True).reset_index()
X_test = pd.merge(X_test, minData.reset_index(), suffixes=["", "_min"], how='left', on=['matchId', 'groupId'])
del minData
X_test = X_test.drop(["heals_min","killStreaks_min","killPoints_min","maxPlace_min","revives_min","headshotKills_min","weaponsAcquired_min","rankPoints_min","matchDuration_min","teamKills_min","numGroups_min","assists_min","roadKills_min","vehicleDestroys_min"], axis=1)
X_test = pd.merge(X_test, minDataRank, suffixes=["", "_minRank"], how='left', on=['matchId', 'groupId'])
del minDataRank
X_test = X_test.drop(["killPoints_minRank","matchDuration_minRank","maxPlace_minRank","numGroups_minRank"], axis=1)


#number of grubs
groupSize = X_test.groupby(['matchId','groupId']).size().reset_index(name='group_size')
X_test = pd.merge(X_test, groupSize, how='left', on=['matchId', 'groupId'])
del groupSize






print(list(X_test.columns).__len__())
print(list(train_data.columns).__len__())
X_test = X_test.drop(['Id','groupId','matchId','kills','damageDealt','numGroups','swimDistance','playerJoined'],axis = 1)




np.shape(X_test),np.shape(test["Id"])
I = np.clip(a = m1.predict(X_test), a_min = 0.0, a_max = 1.0)



submission = pd.DataFrame({
        "Id": test["Id"],
        "winPlacePerc": I
    })
submission.to_csv('submission.csv', index=False)


submission.head()