import pandas as pd
import polars as pl
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
import xgboost as xgb
from sklearn.preprocessing import StandardScaler


train = pd.read_csv("./train_cleaned.csv")
test = pd.read_csv("./test_cleaned.csv")

df = train.copy()
df_test = test.copy()
df_test_1 = pd.read_csv("/project/lt900301-ai24tp/no_hope_5_new.csv")
df_t = df_test.copy()
df_t['nforest_type'] = df_test_1['nforest_type']

df_concat = pd.concat([df, df_t], axis=0, ignore_index=True)
# Shuffling the DataFrame
df_shuffled = df_concat.sample(frac=1).reset_index(drop=True)
df = df_shuffled

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


# Fit the scaler on the training data and transform both training and test data
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)
df_test_scaled = scaler.transform(df_test)


label_encoder = LabelEncoder()
y_balanced = label_encoder.fit_transform(y_balanced)

X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.1, random_state=42)


# Train the final XGBoost model with the best parameters
best_params_xgb = {'n_estimators': 1679, 'max_depth': 13, 'learning_rate': 0.020513663043760136, 'subsample': 0.452963935212717, 'colsample_bytree': 0.49251509118114006, 'min_child_weight': 2}
best_model_xgb = xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss')
best_model_xgb.fit(X_balanced_scaled, y_balanced)


pred = best_model_xgb.predict(df_test_scaled)
pred = label_encoder.inverse_transform(pred)

submission = pd.read_csv("./sample_submission.csv")

submission['nforest_type'][3:] = pred[3:]
submission.to_csv("no_hope_6.csv",index=False)