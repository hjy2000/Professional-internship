# -*- coding: utf-8 -*-
import warnings
warnings.simplefilter('ignore')
import gc
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

from tqdm import tqdm
tqdm.pandas()
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, accuracy_score
from sklearn.metrics import recall_score
import lightgbm as lgb
from sklearn.metrics import f1_score
#预处理

train = pd.read_csv("D:\ASUS\真假职位信息检测/train.csv")
test = pd.read_csv("D:\ASUS\真假职位信息检测/test.csv")

train

test

#把字符串类型的数据转换为小写

for col in ['benefits', 'company_profile', 'department', 'description',
            'employment_type', 'function', 'industry', 'location', 'required_education',
            'required_experience', 'requirements', 'title']:
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()

#处理缺失的数据

def process(x):
    if x == 'nan':
        return 0
    else:
        return len(x.split())


for col in ['benefits', 'title', 'company_profile', 'description', 'requirements']:
    train[f'{col}_wordsLen'] = train[col].astype('str').apply(lambda x: process(x))
    test[f'{col}_wordsLen'] = test[col].astype('str').apply(lambda x: process(x))

def process1(x):
    if x == 'nan':
        return -999
    else:
        try:
            return int(x.split('-')[0])
        except:
            return -998
    
def process2(x):
    if x == 'nan':
        return -999
    else:
        try:
            return int(x.split('-')[1])
        except:
            return -998
    

train['salary_range_start'] = train['salary_range'].astype('str').apply(lambda x: process1(x))
test['salary_range_start'] = test['salary_range'].astype('str').apply(lambda x: process1(x))

train['salary_range_end'] = train['salary_range'].astype('str').apply(lambda x: process2(x))
test['salary_range_end'] = test['salary_range'].astype('str').apply(lambda x: process2(x))

del train['salary_range']
del test['salary_range']

#对具有明显分类的数据添加标签

df = pd.concat([train, test])
del train, test

for f in tqdm(['department', 'employment_type', 'function', 'industry',
               'location', 'required_education', 'required_experience', 'title']):
    lbl = LabelEncoder()
    df[f] = lbl.fit_transform(df[f].astype(str))

train = df[df['fraudulent'].notnull()].copy()
test = df[df['fraudulent'].isnull()].copy()

del df
gc.collect()

#通过TF-IDF处理文本数据

def get_tfidf(train, test, colname, max_features):

    text = list(train[colname].fillna('nan').values)
    tf = TfidfVectorizer(min_df=0, 
                         ngram_range=(1,2), 
                         stop_words='english', 
                         max_features=max_features)
    tf.fit(text)
    X = tf.transform(text)
    X_test = tf.transform(list(test[colname].fillna('nan').values))

    df_tfidf = pd.DataFrame(X.todense())
    df_tfidf_test = pd.DataFrame(X_test.todense())
    df_tfidf.columns = [f'{colname}_tfidf{i}' for i in range(max_features)]
    df_tfidf_test.columns = [f'{colname}_tfidf{i}' for i in range(max_features)]
    for col in df_tfidf.columns:
        train[col] = df_tfidf[col]
        test[col] = df_tfidf_test[col]
        
    return train, test


train, test = get_tfidf(train, test, 'benefits', 12)
train, test = get_tfidf(train, test, 'company_profile', 24)
train, test = get_tfidf(train, test, 'description', 48)
train, test = get_tfidf(train, test, 'requirements', 20)

to_drop = ['benefits', 'company_profile', 'description', 'requirements']

train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)

train['id'] = train.index
test['id'] = test.index

train.shape, test.shape

train.head()

#模型训练

ycol = 'fraudulent'#预测结果
feature_names = list(
    filter(lambda x: x not in [ycol, 'id'], train.columns))

model = lgb.LGBMClassifier(objective='binary',
                           boosting_type='gbdt',
                           tree_learner='serial',
                           num_leaves=32,#叶子节点数
                           max_depth=6,#最大树深
                           learning_rate=0.1,#学习率
                           n_estimators=10000,#决策树个数
                           subsample=0.8,
                           feature_fraction=0.6,#特征分数
                           reg_alpha=10,
                           reg_lambda=12,
                           random_state=1983,
                           is_unbalance=True,
                           metric='auc')


oof = []
prediction = test[['id']]
prediction['fraudulent'] = 0
df_importance_list = []

SKfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1983)
for fold_id, (trn_idx, val_idx) in enumerate(SKfold.split(train[feature_names], train[ycol])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][ycol]

    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][ycol]

    print('\nFold_{} Training ===================\n'.format(fold_id+1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',#评价标准
                          early_stopping_rounds=50)   #fit

    pred_val = lgb_model.predict(   #预测
        X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['id', ycol]].copy()
    df_oof['pred'] = pred_val
    oof.append(df_oof)

    pred_test = lgb_model.predict(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction['fraudulent'] += pred_test / SKfold.n_splits

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,   #重要性
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()

#查看元素数据各个元素的重要性

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg(
    'mean').sort_values(ascending=False).reset_index()
print(df_importance)


#lgb模型得分

df_oof = pd.concat(oof)
print()
score = accuracy_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'))
print('auc:', score)
print()
pScore=precision_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'),average='macro')
print('精确率：',pScore)
print()
recall=recall_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'),average='macro')
print('召回率：',recall)
print()
f1_score=f1_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'),average='macro')
print('f1值：',f1_score)
print()
#另一个F1计算方法
p_class, r_class, f_class, support_micro= precision_recall_fscore_support(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'))
#print('class:', p_class, r_class, f_class, support_micro)
print('各类单独F1:',f_class)
print('各类F1取平均：',f_class.mean())
print()
prediction.fraudulent.value_counts()

#对预测结果进行处理

sub = prediction.copy(deep=True)
sub['fraudulent'] = sub['fraudulent'].apply(lambda x: 1 if x==1 else 0)

print('分类个数相关情况')
print(sub.fraudulent.value_counts())

sub.to_csv('./submissions.csv'.format(score), index=False, header=False, encoding='utf-8')

