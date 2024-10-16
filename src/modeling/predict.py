from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import joblib
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR

def predict_random_forests(df: pd.DataFrame):
    """Loads the latest RandomForestRegression model and predicts the r_spread for given input.

    Args:
        df (pd.DataFrame): Dataframe formatted to input_struct specification (See data/db/input_struct)

    Returns:
        array: Array of r_spread's for every game in df
    """
    rfr_path = MODELS_DIR / "rfr_latest.pkl"
    rfr = joblib.load(rfr_path)
    return rfr.predict(df)

def predict_support_vectors(df: pd.DataFrame):
    """Loads the latest SupportVectorRegression model and predicts the r_spread for given input.

    Args:
        df (pd.DataFrame): Dataframe formatted to input_struct specification (See data/db/input_struct)

    Returns:
        array: Array of r_spread's for every game in df
    """
    svr_path = MODELS_DIR / "svr_latest.pkl"
    svr = joblib.load(svr_path)
    return svr.predict(df)

def predict_gbt(df: pd.DataFrame):
    """Loads the latest SupportVectorRegression model and predicts the r_spread for given input.

    Args:
        df (pd.DataFrame): Dataframe formatted to input_struct specification (See data/db/input_struct)

    Returns:
        array: Array of r_spread's for every game in df
    """
    gbt_path = MODELS_DIR / "gbt_latest.pkl"
    gbt = joblib.load(gbt_path)
    return gbt.predict(df)

if __name__ == "__main__":
    logger.info("predictions require input")

