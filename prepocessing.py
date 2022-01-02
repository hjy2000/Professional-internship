import warnings
warnings.simplefilter('ignore')
import gc
import numpy as np
import pandas as pd
pd.set_option('max_columns',None)
pd.set_option('max_rows',None)
from tqdm import tqdm
tqdm.pandas()

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import auc,accuracy_score

import lightgbm as lgb

##预处理部分
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#把字符串类型转换为小写
for col in ['benefits','company_profile','department','description',
            'employment_type','function','industry','location','required_education',
            'required_experience','requirements','title']:
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()

#处理缺失的数据
def process(x):
    if x == 'nan':
        return 0
    else:
        return len(x.split())

for col in ['benefits', 'title', 'company_profile',
            'description', 'requirements']:
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
def get_tfidf(train,test,colname,max_features):
    text = list(train[colname].fillna('nan').values)
    tf = TfidfVectorizer(min_df=0,ngram_range=(1,2),stop_words='english',
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

    return train,test

train,test = get_tfidf(train,test,'benefits',12)
train,test = get_tfidf(train,test,'company_profile',24)
train,test = get_tfidf(train,test,'description',48)
train,test = get_tfidf(train,test,'requirements',20)


to_drop = ['benefits', 'company_profile', 'description', 'requirements']
train = train.drop(to_drop, axis=1)
test = test.drop(to_drop, axis=1)

train['id'] = train.index
test['id'] = test.index
train.shape, test.shape

train.head()

##以下可开始进行模型训练