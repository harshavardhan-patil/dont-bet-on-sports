from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import typer
from loguru import logger
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR

rfr_path = MODELS_DIR / "rfr" / "rfr_latest.pkl"
def predict_random_forests(df: pd.DataFrame):
    """Loads the latest RandomForestRegression model and predicts the r_spread for given input.

    Args:
        df (pd.DataFrame): Dataframe formatted to input_struct specification (See data/db/input_struct)

    Returns:
        array: Array of r_spread's for every game in df
    """
    rfr = joblib.load(rfr_path)
    return rfr.predict(df)

if __name__ == "__main__":
    logger.info("predictions require input")

