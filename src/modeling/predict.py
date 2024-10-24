from loguru import logger
from src.config import PROCESSED_DATA_DIR
import torch
from src.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, MODELS_DIR
from src.modeling.nn.neuralnets import NFLPredictor
from pathlib import Path
import pandas as pd
import joblib

from src.config import MODELS_DIR

scaler_path = MODELS_DIR / "scalers" / "scaler_y.gz"

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
    results = svr.predict(df)
    #using unscaled spread values
    scaler_y = joblib.load(scaler_path)
    return scaler_y.inverse_transform(results.reshape(-1, 1))

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

def predict_nn(df: pd.DataFrame):
    X =  torch.tensor(df.values, dtype=torch.float32)
    nn_path = MODELS_DIR / "neural_net.pt"
    input_dim = X.shape[1]
    model = NFLPredictor(input_dim)
    model.load_state_dict(torch.load(str(nn_path), weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    results = None
    with torch.no_grad():
        results = model(X)
    #using unscaled spread values
    scaler_y = joblib.load(scaler_path)
    return scaler_y.inverse_transform(results.reshape(-1, 1))

if __name__ == "__main__":
    logger.info("predictions require input")

