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

from dbos.config import MODELS_DIR, PROCESSED_DATA_DIR

processed_train_path: Path = PROCESSED_DATA_DIR / "trainset.csv"
rfr_base_path = MODELS_DIR / "rfr"

def train_random_forests(v_update: bool = False):
    df = pd.read_csv(str(processed_train_path), index_col='id')
    df.head()

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    rfr = RandomForestRegressor()
    rfr.fit(X, y)

    #todo - add versioning with v_update
    rfr_output_path = rfr_base_path / "rfr_latest.pkl"
    joblib.dump(rfr, rfr_output_path) 

if __name__ == "__main__":
    train_random_forests()
