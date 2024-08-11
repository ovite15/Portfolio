import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score ,log_loss
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from collections import Counter
from catboost import CatBoostClassifier, Pool
import optuna
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

## preprocess 
## This train and test add more features 
train = pd.read_csv("./train_cleaned.csv")
test = pd.read_csv("./test_cleaned.csv")
df = train.copy()
df_test = test.copy()

# Make new feature
df['Modified Simple Ratio'] = (df['b8'] - df['b1']) / (df['b4'] - df['b1'])
df_test['Modified Simple Ratio'] = (df_test['b8'] - df_test['b1']) / (df_test['b4'] - df_test['b1'])

# Handle inf value with mean value
df['Modified Simple Ratio']
mean_value = df['Modified Simple Ratio'].replace([np.inf, -np.inf], np.nan).mean()
df['Modified Simple Ratio'].replace([np.inf, -np.inf], mean_value, inplace=True)
df_test['Modified Simple Ratio']
mean_value = df_test['Modified Simple Ratio'].replace([np.inf, -np.inf], np.nan).mean()
df_test['Modified Simple Ratio'].replace([np.inf, -np.inf], mean_value, inplace=True)

# Set name that enable to model
df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
df_test.columns = df_test.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

X = df.drop('nforest_type', axis=1)
y = df['nforest_type']

# Handle class imbalance using RandomOverSampler
counter_before = Counter(y)
print('Before:', counter_before)

oversampler = RandomOverSampler(random_state=42)
X_balanced, y_balanced = oversampler.fit_resample(X, y)

counter_before = Counter(y_balanced)
print('Before:', counter_before)

# Standardize
scaler = StandardScaler()
X_balanced_scaled = scaler.fit_transform(X_balanced)
df_test_scaled = scaler.transform(df_test)
label_encoder = LabelEncoder()
y_balanced = label_encoder.fit_transform(y_balanced)

# To train model XGB
X_train, X_test, y_train, y_test = train_test_split(X_balanced_scaled, y_balanced, test_size=0.1, random_state=42)

# Define the objective function for XGBoost
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'multi:softmax',
        'num_class': len(set(y))
    }


    model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=100, verbose=False)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

# Run the optimization for XGBoost
study_xgb = optuna.create_study(direction='maximize')
study_xgb.optimize(objective_xgb, n_trials=50)

print('XGBoost - Number of finished trials:', len(study_xgb.trials))
print('XGBoost - Best trial:', study_xgb.best_trial.params)

# Train the final XGBoost model with the best parameters
best_params_xgb = study_xgb.best_trial.params
best_model_xgb = xgb.XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='mlogloss')
best_model_xgb.fit(X_balanced_scaled, y_balanced)

pred = best_model_xgb.predict(df_test_scaled)
pred = label_encoder.inverse_transform(pred)
submission = pd.read_csv("./sample_submission.csv")

submission['nforest_type'][3:] = pred[3:]
submission.to_csv("no_hope_5.csv",index=False)

