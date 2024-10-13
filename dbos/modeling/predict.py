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

from dbos.config import MODELS_DIR

rfr_path = MODELS_DIR / "rfr" / "rfr_latest.pkl"
def predict_random_forests(df: pd.DataFrame):
    rfr = joblib.load(rfr_path)
    return rfr.predict(df)

'''if __name__ == "__main__":
    logger.info("import guard")'''

