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
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    rfr = RandomForestRegressor()
    rfr.fit(X, y)

    rfr_output_path = MODELS_DIR / "rfr_latest.pkl"
    joblib.dump(rfr, rfr_output_path) 

def train_support_vectors():
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    svr = SVR(kernel='rbf')
    svr.fit(X, y)

    svr_output_path = MODELS_DIR / "svr_latest.pkl"
    joblib.dump(svr, svr_output_path) 

def train_gbt():
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    gbt = xgb.XGBRegressor()
    gbt.fit(X, y)

    gbt_output_path = MODELS_DIR / "gbt_latest.pkl"
    joblib.dump(gbt, gbt_output_path) 

if __name__ == "__main__":
    train_gbt()
