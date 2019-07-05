# -*- coding: utf-8 -*-
"""
@author: shaowu

注：此次会详细注释代码，往后都省略。
"""
import pandas as pd
import numpy as np
import time
import tqdm
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from collections import Counter
def one_hot_col(col):
    '''标签编码'''
    lbl = preprocessing.LabelEncoder()
    lbl.fit(col)
    return lbl
def calculate_null(data,key,col):
    '''
    params:
    data -- input data
    key -- the key used for statistics
    col -- the columns for statistics
    return -- the data of DataFrame type, include two columns,
              first columns id key,second is number of null
    '''
    return data.groupby(key,as_index=False)[col].agg({col+'_is_null':'count'})
def xgb_model(new_train,y,new_test,lr):
    '''定义模型'''
    xgb_params = {'booster': 'gbtree',
          'eta':lr, 'max_depth': 5, 'subsample': 0.8, 'colsample_bytree': 0.8, 
          'objective':'binary:logistic',
          'eval_metric': 'auc',
          'silent': True,
          }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_xgb=np.zeros(new_train.shape[0])
    prediction_xgb=np.zeros(new_test.shape[0])
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = xgb.DMatrix(new_train[tr],y[tr])
        dvalid = xgb.DMatrix(new_train[va],y[va])
        watchlist = [(dtrain, 'train'), (dvalid, 'valid_data')]
        bst = xgb.train(dtrain=dtrain, num_boost_round=30000, evals=watchlist, early_stopping_rounds=200, \
        verbose_eval=50, params=xgb_params)
        oof_xgb[va] += bst.predict(xgb.DMatrix(new_train[va]), ntree_limit=bst.best_ntree_limit)
        prediction_xgb += bst.predict(xgb.DMatrix(new_test), ntree_limit=bst.best_ntree_limit)
    print('the roc_auc_score for train:',roc_auc_score(y,oof_xgb))
    prediction_xgb/=5
    return oof_xgb,prediction_xgb
def lgb_model(new_train,y,new_test):
    params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_leaves': 1000,
    'verbose': -1,
    'max_depth': -1,
  #  'reg_alpha':2.2,
  #  'reg_lambda':1.4,
    'seed':42,
    }
    #skf=StratifiedKFold(y,n_folds=5,shuffle=True,random_state=2018)
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    oof_lgb=np.zeros(new_train.shape[0]) ##用于存放训练集概率，由每折验证集所得
    prediction_lgb=np.zeros(new_test.shape[0])  ##用于存放测试集概率，k折最后要除以k取平均
    feature_importance_df = pd.DataFrame() ##存放特征重要性，此处不考虑
    for i,(tr,va) in enumerate(skf.split(new_train,y)):
        print('fold:',i+1,'training')
        dtrain = lgb.Dataset(new_train[tr],y[tr])
        dvalid = lgb.Dataset(new_train[va],y[va],reference=dtrain)
        ##训练：
        bst = lgb.train(params, dtrain, num_boost_round=30000, valid_sets=dvalid, verbose_eval=400,early_stopping_rounds=200)
        ##预测验证集：
        oof_lgb[va] += bst.predict(new_train[va], num_iteration=bst.best_iteration)
        ##预测测试集：
        prediction_lgb += bst.predict(new_test, num_iteration=bst.best_iteration)
        '''
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = list(new_train.columns)
        fold_importance_df["importance"] = bst.feature_importance(importance_type='split', iteration=bst.best_iteration)
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        '''
    
    print('the roc_auc_score for train:',roc_auc_score(y,oof_lgb)) ##线下auc评分
    prediction_lgb/=5
    return oof_lgb,prediction_lgb,feature_importance_df

##读入测试数据：
testdata= pd.read_csv("round1_iflyad_anticheat_testdata_feature.txt",sep='\t')
testdata['label']=-1 ##测试集没有标签，可标记为-1

testdata['begin_time']=testdata['sid'].apply(lambda x:int(x.split('-')[-1])) ##请求会话时间
testdata['nginxtime-begin_time']=testdata['nginxtime']-testdata['begin_time'] ##请求会话时间 与 请求到达服务时间的差

##读入训练数据：
traindata= pd.read_csv("round1_iflyad_anticheat_traindata.txt",sep='\t')

traindata['begin_time']=traindata['sid'].apply(lambda x:int(x.split('-')[-1]))
traindata['nginxtime-begin_time']=traindata['nginxtime']-traindata['begin_time']

##结合数据，方便提取特征：axis=0 纵向合并；axis=1 横向合并
data=pd.concat([traindata,testdata],axis=0).reset_index(drop=True)

print('the shape of data:',data.shpe)

print(data.nunique()) ##返回每个字段的所有值组成集合的大小，即集合元素个数
print(data[:5]) ##输出数据前5行
z=calculate_null(testdata,'sid','ver') ##计算缺失值的，下面还没用到

print('label distribution:\n',traindata['label'].value_counts()) ##查看训练集标签分布

object_cols=list(data.dtypes[data.dtypes==np.object].index) ##返回字段名为object类型的字段
print(data.dtypes[data.dtypes==np.object].index) ##输出object类型的字段

##本题所给时间戳为毫秒级，故需除以1000转换为秒级：时间戳转成日期格式
print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(data['nginxtime'][0]/1000)))

##对object类型的字段进行标签编码：
for col in object_cols:
    if col!='sid':
        data[col]=one_hot_col(data[col].astype(str)).transform(data[col].astype(str))

##划分数据：
train=data[:traindata.shape[0]]
label=train['label'].values
test=data[traindata.shape[0]:].reset_index(drop=True)

##模型训练预测：
oof_lgb,prediction_lgb,feature_importance_df=\
      lgb_model(np.array(train.drop(['sid','label','nginxtime','ip','reqrealip','begin_time'],axis=1)),\
                label,\
                np.array(test.drop(['sid','label','nginxtime','ip','reqrealip','begin_time'],axis=1)))

##保存结果：
sub=test[['sid']]
sub['label']=prediction_lgb
sub['label']=sub['label'].apply(lambda x: 1 if x>0.5 else 0) ##∪概率大于0.5的置1，否则置0
print('test pre_label distribution:\n',sub['label'].value_counts()) ## 模型预测测试集的标签分布
sub.to_csv('submit0704.csv',index=None) ##保存为submit0704.csv文件

