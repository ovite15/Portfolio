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
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)


def objective(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 10000),  # Increase the range for more iterations
        "depth": trial.suggest_int("depth", 4, 12),  # Slightly wider range for tree depth
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 1e-1),  # Smaller minimum learning rate
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-4, 1e2),  # Wider range for L2 regularization
        "bagging_temperature": trial.suggest_loguniform("bagging_temperature", 0.01, 10.0),
        "random_strength": trial.suggest_loguniform("random_strength", 1e-4, 10.0),
        # "rsm": trial.suggest_uniform("rsm", 0.5, 1.0),  # Random subspace method, similar to feature bagging
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    # Create Pool object for CatBoost
    train_pool = Pool(X_train, y_train)
    test_pool = Pool(X_test, y_test)

    # Initialize and train the CatBoost model
    model = CatBoostClassifier(**params, task_type="GPU", devices='0',silent=True)
    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=100)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)  # Increase the number of trials for a more thorough search

print(f"Best trial: {study.best_trial.params}")

from catboost import CatBoostClassifier, Pool

parmas = {'iterations': 3122, 'depth': 8, 'learning_rate': 0.06957441515739986, 'l2_leaf_reg': 0.00037234418819357925, 'bagging_temperature': 0.07731097029760352, 'random_strength': 0.0005299812317460983, 'min_data_in_leaf': 63}
CatModel = CatBoostClassifier(**study.best_trial.params)

# Fit the model
CatModel.fit(X_balanced, y_balanced, early_stopping_rounds=100,plot=True)
Cat_pred = CatModel.predict(df_test)

submission = pd.read_csv("./sample_submission.csv")

submission['nforest_type'][3:] = Cat_pred[3:]
submission.to_csv("no_hope_1.csv",index=False)
