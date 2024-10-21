from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import typer
#from loguru import logger
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
import xgboost as xgb

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

processed_train_path: Path = PROCESSED_DATA_DIR / "trainset.csv"

def train_random_forests():
    '''tuned hyperparams: 'n_estimators': 950, 'min_weight_fraction_leaf': 0.125, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'ccp_alpha': 0.07777777777777778, 'bootstrap': True'''
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    rfr = RandomForestRegressor(n_estimators=950, 
                                min_weight_fraction_leaf = 0.125,
                                min_samples_split = 5,
                                min_samples_leaf=1,
                                max_depth=30,
                                ccp_alpha=0.07777777777777778,
                                bootstrap=True,)
    rfr.fit(X, y)

    rfr_output_path = MODELS_DIR / "rfr_latest.pkl"
    joblib.dump(rfr, rfr_output_path) 

def train_support_vectors():
    '''tuned hyperparams: 'kernel': 'rbf', 'gamma': 0.001, 'epsilon': 0.12, 'C': 7.742636826811277'''
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    svr = SVR(kernel='rbf',
              gamma= 0.001, 
              epsilon=0.12, 
              C=7.742636826811277)
    svr.fit(X, y)

    svr_output_path = MODELS_DIR / "svr_latest.pkl"
    joblib.dump(svr, svr_output_path) 

def train_gbt():
    '''tuned hyperparams: 'subsample': 0.8333333333333333, 'reg_lambda': 0.1, 'reg_alpha': 0.001, 'n_estimators': 200, 'learning_rate': 0.036000000000000004, 'gamma': 0.25, 'colsample_bytree': 0.5}'''
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    gbt = xgb.XGBRegressor(objective='reg:absoluteerror', random_state=42, n_jobs=-1, 
                           subsample=0.8333333333333333, 
                           reg_lambda=0.1, 
                           reg_alpha=0.001, 
                           n_estimators=200, 
                           learning_rate=0.036000000000000004, 
                           gamma=0.25,
                           colsample_bytree=0.5,)
    gbt.fit(X, y)

    gbt_output_path = MODELS_DIR / "gbt_latest.pkl"
    joblib.dump(gbt, gbt_output_path) 

if __name__ == "__main__":
    train_random_forests()
