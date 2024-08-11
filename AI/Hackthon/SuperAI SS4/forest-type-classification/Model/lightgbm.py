import pandas as pd
import polars as pl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import Counter
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
train = pd.read_csv("./train_cleaned.csv")
test = pd.read_csv("./test_cleaned.csv")

df = train.copy()
df_test = test.copy()

df['Modified Simple Ratio'] = (df['b8'] - df['b1']) / (df['b4'] - df['b1'])
df_test['Modified Simple Ratio'] = (df_test['b8'] - df_test['b1']) / (df_test['b4'] - df_test['b1'])

df['Modified Simple Ratio']
mean_value = df['Modified Simple Ratio'].replace([np.inf, -np.inf], np.nan).mean()
df['Modified Simple Ratio'].replace([np.inf, -np.inf], mean_value, inplace=True)

df_test['Modified Simple Ratio']
mean_value = df_test['Modified Simple Ratio'].replace([np.inf, -np.inf], np.nan).mean()
df_test['Modified Simple Ratio'].replace([np.inf, -np.inf], mean_value, inplace=True)

df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df_test.columns = df_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

X = df.drop('nforest_type', axis=1)
y = df['nforest_type']

counter_before = Counter(y)
print('Before:', counter_before)

oversampler = RandomOverSampler(random_state=42)
X_balanced, y_balanced = oversampler.fit_resample(X, y)

counter_before = Counter(y_balanced)
print('Before:', counter_before)
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test data
X_balanced_scaled = scaler.fit_transform(X_balanced)
df_test_scaled = scaler.transform(df_test)

label_encoder = LabelEncoder()
y_balanced = label_encoder.fit_transform(y_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.1, random_state=42)

train_real_data = lgb.Dataset(X_balanced_scaled, label=y_balanced)
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'num_leaves': 119,
    'learning_rate': 0.08638608659491388,
    'feature_fraction': 0.9212076153108513,
    'bagging_fraction': 0.7576210730193799,
    'bagging_freq': 4,
    'min_child_samples': 54,
    'lambda_l1': 1.3333758596685116e-08,
    'lambda_l2': 0.0015080069557647277,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': len(np.unique(y_balanced)),  # Number of classes
    'boosting_type': 'gbdt'
}

gbm = lgb.train(params,
                train_data,
                num_boost_round=100,
                valid_sets=[train_data, test_data])

y_pred_probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)
y_pred = np.argmax(y_pred_probs, axis=1)

print(accuracy_score(y_test , y_pred))

gbm = lgb.train(params,
                train_real_data,
                num_boost_round=100)

pred_probs = gbm.predict(df_test_scaled)
pred = np.argmax(pred_probs, axis=1)
pred = label_encoder.inverse_transform(pred)

submission = pd.read_csv("./sample_submission.csv")

submission['nforest_type'][3:] = pred[3:]
submission.to_csv("no_hope_8.csv",index=False)