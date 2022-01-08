from flask import Flask, render_template, request, jsonify

import warnings
import gc
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from joblib import dump, load
from flask_cors import *

app = Flask(__name__)
CORS(app, supports_credentials=True)

warnings.simplefilter('ignore')
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)
tqdm.pandas()


# 主页
@app.route('/')
def test():
    return render_template('test.html')


# 接收各数据元素重要性图请求
@app.route('/g2', methods=["post"])
def g2():
    importance = pd.read_csv('./importance.csv', header=None)
    return jsonify(importance.to_json(orient='split'))


@app.route('/g1', methods=["post"])
def g1():
    prediction = pd.read_csv('./prediction.csv', header=None)
    return jsonify(prediction.to_json(orient='split'))


# 接收文件并进行处理
@app.route('/receive', methods=["post"])
def receive():
    # train = request.files.get("train")
    test = request.files['file']
    # if train and test:
    #     saveModel(pd.read_csv(train), pd.read_csv(test))
    #     return render_template('test.html')
    if test:
        prediction = predict(pd.read_csv(test))
        return jsonify(prediction.to_json(orient='split'))
    # return render_template('test.html')


# 预测test集结果
def predict(test):
    """对test集进行预处理"""
    """把字符串类型的数据转换为小写"""
    for col in ['benefits', 'company_profile', 'department', 'description',
                'employment_type', 'function', 'industry', 'location', 'required_education',
                'required_experience', 'requirements', 'title']:
        test[col] = test[col].str.lower()

    """处理缺失的数据"""
    for col in ['benefits', 'title', 'company_profile', 'description', 'requirements']:
        test[f'{col}_wordsLen'] = test[col].astype('str').apply(lambda x: process(x))

    test['salary_range_start'] = test['salary_range'].astype('str').apply(lambda x: process1(x))

    test['salary_range_end'] = test['salary_range'].astype('str').apply(lambda x: process2(x))

    del test['salary_range']

    """对具有明显分类的数据添加标签"""
    test['fraudulent'] = None

    for f in tqdm(['department', 'employment_type', 'function', 'industry',
                   'location', 'required_education', 'required_experience', 'title']):
        lbl = LabelEncoder()
        test[f] = lbl.fit_transform(test[f].astype(str))

    test = test[test['fraudulent'].isnull()].copy()

    gc.collect()

    """通过TF-IDF处理文本数据"""
    test = get_test_tfidf(test, 'benefits', 12)
    test = get_test_tfidf(test, 'company_profile', 24)
    test = get_test_tfidf(test, 'description', 48)
    test = get_test_tfidf(test, 'requirements', 20)

    to_drop = ['benefits', 'company_profile', 'description', 'requirements']

    test = test.drop(to_drop, axis=1)

    test['id'] = test.index

    """载入训练好的模型"""
    lgb_model = load('./lgb_model.pkl')

    """预测结果"""
    ycol = 'fraudulent'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'id'], test.columns))
    pred_test = lgb_model.predict(test[feature_names])
    prediction = [pred_test.size - sum(pred_test == 0), sum(pred_test == 0)]
    prediction = pd.DataFrame(prediction)
    prediction.to_csv('./prediction.csv', index=False, header=False, encoding='utf-8')
    return prediction

    # prediction = test[['id']]
    # prediction['fraudulent'] = pred_test
    # prediction['fraudulent'] = prediction['fraudulent'].apply(lambda x: 1 if x == 1 else 0)


# 训练并保存模型
def saveModel(train, test):
    """把字符串类型的数据转换为小写"""
    for col in ['benefits', 'company_profile', 'department', 'description',
                'employment_type', 'function', 'industry', 'location', 'required_education',
                'required_experience', 'requirements', 'title']:
        train[col] = train[col].str.lower()
        test[col] = test[col].str.lower()

    """处理缺失的数据"""
    for col in ['benefits', 'title', 'company_profile', 'description', 'requirements']:
        train[f'{col}_wordsLen'] = train[col].astype('str').apply(lambda x: process(x))
        test[f'{col}_wordsLen'] = test[col].astype('str').apply(lambda x: process(x))

    train['salary_range_start'] = train['salary_range'].astype('str').apply(lambda x: process1(x))
    test['salary_range_start'] = test['salary_range'].astype('str').apply(lambda x: process1(x))

    train['salary_range_end'] = train['salary_range'].astype('str').apply(lambda x: process2(x))
    test['salary_range_end'] = test['salary_range'].astype('str').apply(lambda x: process2(x))

    del train['salary_range']
    del test['salary_range']

    """对具有明显分类的数据添加标签"""
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

    """通过TF-IDF处理文本数据"""
    train, test = get_tfidf(train, test, 'benefits', 12)
    train, test = get_tfidf(train, test, 'company_profile', 24)
    train, test = get_tfidf(train, test, 'description', 48)
    train, test = get_tfidf(train, test, 'requirements', 20)

    to_drop = ['benefits', 'company_profile', 'description', 'requirements']

    train = train.drop(to_drop, axis=1)
    test = test.drop(to_drop, axis=1)

    train['id'] = train.index
    test['id'] = test.index

    train.head()

    """## 模型训练"""
    ycol = 'fraudulent'
    feature_names = list(
        filter(lambda x: x not in [ycol, 'id'], train.columns))
    print(feature_names)

    model = lgb.LGBMClassifier(objective='binary',
                               boosting_type='gbdt',
                               tree_learner='serial',
                               num_leaves=32,
                               max_depth=6,
                               learning_rate=0.1,
                               n_estimators=10000,
                               subsample=0.8,
                               feature_fraction=0.6,
                               reg_alpha=10,
                               reg_lambda=12,
                               random_state=1983,
                               is_unbalance=True,
                               metric='auc')

    oof = []
    prediction = test[['id']]
    prediction['fraudulent'] = 0
    df_importance_list = []
    best_model = model
    best_acc = 0

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1983)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[ycol])):
        X_train = train.iloc[trn_idx][feature_names]
        Y_train = train.iloc[trn_idx][ycol]

        X_val = train.iloc[val_idx][feature_names]
        Y_val = train.iloc[val_idx][ycol]

        # print('\nFold_{} Training ================================\n'.format(fold_id + 1))

        lgb_model = model.fit(X_train,
                              Y_train,
                              eval_names=['train', 'valid'],
                              eval_set=[(X_train, Y_train), (X_val, Y_val)],
                              verbose=500,
                              eval_metric='auc',
                              early_stopping_rounds=50)

        pred_val = lgb_model.predict(
            X_val, num_iteration=lgb_model.best_iteration_)
        df_oof = train.iloc[val_idx][['id', ycol]].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        acc = accuracy_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'))
        if acc > best_acc:
            best_acc = acc
            best_model = lgb_model

        pred_test = lgb_model.predict(
            test[feature_names], num_iteration=lgb_model.best_iteration_)
        prediction['fraudulent'] += pred_test / kfold.n_splits

        df_importance = pd.DataFrame({
            'column': feature_names,
            'importance': lgb_model.feature_importances_,
        })
        df_importance_list.append(df_importance)

        del pred_val, pred_test, X_train, Y_train, X_val, Y_val
        gc.collect()

    """保存模型"""
    dump(best_model, './lgb_model.pkl')

    """查看元素数据各个元素的重要性"""
    df_importance = pd.concat(df_importance_list)
    df_importance = df_importance.groupby(['column'])['importance'].agg(
        'mean').sort_values(ascending=False).reset_index()
    df_importance.to_csv('./importance.csv', index=False, header=False, encoding='utf-8')
    # return df_importance.to_json(orient='split')

    # """获得lgb模型训练后的得分"""
    # df_oof = pd.concat(oof)
    #
    # score = accuracy_score(df_oof[ycol].astype('int'), df_oof['pred'].astype('int'))
    # # print('auc:', score)
    #
    # prediction.fraudulent.value_counts()
    #
    # """对预测结果进行处理"""
    # sub = prediction.copy(deep=True)
    # sub['fraudulent'] = sub['fraudulent'].apply(lambda x: 1 if x == 1 else 0)
    #
    # # print(sub.fraudulent.value_counts())
    # sub.to_csv('./submissions.csv'.format(score), index=False, header=False, encoding='utf-8')
    # return sub.to_json(orient='split')


def process(x):
    if x == 'nan':
        return 0
    else:
        return len(x.split())


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


def get_tfidf(train, test, colname, max_features):
    text = list(train[colname].fillna('nan').values)
    tf = TfidfVectorizer(min_df=0,
                         ngram_range=(1, 2),
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


def get_test_tfidf(test, colname, max_features):
    text = list(test[colname].fillna('nan').values)
    tf = TfidfVectorizer(min_df=0,
                         ngram_range=(1, 2),
                         stop_words='english',
                         max_features=max_features)
    tf.fit(text)
    X_test = tf.transform(text)

    df_tfidf_test = pd.DataFrame(X_test.todense())
    df_tfidf_test.columns = [f'{colname}_tfidf{i}' for i in range(max_features)]
    for col in df_tfidf_test.columns:
        test[col] = df_tfidf_test[col]

    return test


if __name__ == '__main__':
    app.run()
