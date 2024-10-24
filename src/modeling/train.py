from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
import xgboost as xgb
from src.modeling.nn.neuralnets import NFLPredictor

from src.config import MODELS_DIR, PROCESSED_DATA_DIR

scaled_train_path: Path = PROCESSED_DATA_DIR / "scaled" / "trainset.csv"
scaled_test_path: Path = PROCESSED_DATA_DIR / "scaled" /"testset.csv"
unscaled_train_path: Path = PROCESSED_DATA_DIR / "unscaled" / "trainset.csv"
unscaled_test_path: Path = PROCESSED_DATA_DIR / "unscaled" /"testset.csv"

def train_random_forests():
    '''tuned hyperparams: 'n_estimators': 950, 'min_weight_fraction_leaf': 0.125, 'min_samples_split': 5, 'min_samples_leaf': 1, 'max_depth': 30, 'ccp_alpha': 0.07777777777777778, 'bootstrap': True'''
    df = pd.read_csv(str(unscaled_train_path), index_col='id')

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
    logger.info("Random Forests Model trained")

def train_support_vectors():
    '''tuned hyperparams: 'kernel': 'rbf', 'gamma': 0.001, 'epsilon': 0.12, 'C': 7.742636826811277'''
    df = pd.read_csv(str(scaled_train_path), index_col='id')

    X, y = df.drop(columns=['r_spread']), df.loc[:,'r_spread']
    svr = SVR(kernel='rbf',
              gamma= 0.001, 
              epsilon=0.12, 
              C=7.742636826811277)
    svr.fit(X, y)

    svr_output_path = MODELS_DIR / "svr_latest.pkl"
    joblib.dump(svr, svr_output_path) 
    logger.info("Support Vector Machines Model trained")

def train_gbt():
    '''tuned hyperparams: 'subsample': 0.8333333333333333, 'reg_lambda': 0.1, 'reg_alpha': 0.001, 'n_estimators': 200, 'learning_rate': 0.036000000000000004, 'gamma': 0.25, 'colsample_bytree': 0.5}'''
    df = pd.read_csv(str(unscaled_train_path), index_col='id')

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
    logger.info("Gradient Boosted Trees Model trained")

def train_nn():
    df = pd.read_csv(str(scaled_train_path), index_col='id')
    X, y = df.drop(columns=['r_spread']).values, df.loc[:,'r_spread'].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device to train NN: {device}")

    X = torch.tensor(X, dtype=torch.float32).to(device=device)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device=device)

    # Initialize the model
    input_dim = X.shape[1]
    model = NFLPredictor(input_dim).to(device=device)

    # Define loss function and optimizer
    criterion = nn.L1Loss()  # Using MAE
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # Training loop with batch training
    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(X, y), batch_size=16, shuffle=True
    )

    epochs = 100
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        # Adjust learning rate with scheduler
        scheduler.step(loss)
        from src.config import MODELS_DIR

    nn_path = MODELS_DIR / "neural_net.pt"
    torch.save(model.state_dict(), str(nn_path))
    logger.info("Neural Network trained")

if __name__ == "__main__":
    train_random_forests()
