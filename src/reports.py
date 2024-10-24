from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import mean_absolute_error 
from src.config import FIGURES_DIR, INTERIM_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
import joblib
from src.modeling.predict import predict_random_forests, predict_support_vectors, predict_gbt, predict_nn

#toggle for saving new plots, if false plots are only displayed
SAVE = True

RF = "Random Forests"
SVM = "Support Vector Machines"
GBT = "Gradient Boosted Trees"
NN = "Neural Network"

vegas_path: Path = INTERIM_DATA_DIR / 'testset.csv'
scaled_test_path: Path = PROCESSED_DATA_DIR / "scaled" /"testset.csv"
unscaled_test_path: Path = PROCESSED_DATA_DIR / "unscaled" /"testset.csv"

df_scaled = pd.read_csv(str(scaled_test_path), index_col='id')
df_unscaled = pd.read_csv(str(unscaled_test_path), index_col='id')

def analyze_random_forests():
    ''' For trained Random Forests:
    1. Calculate MAE
    2. Plot comparison with Vegas
    3. Plot Betting Simulation
    '''
    results = predict_random_forests(df_unscaled.drop(columns=['r_spread']))
    mae = mean_absolute_error(df_unscaled['r_spread'], results)
    logger.info(f"{RF} MAE: {mae}")
    df_model = df_unscaled.assign(pred = results)
    plot_error(df_model, RF)
    plot_sim(results, RF, 2)

def analyze_support_vectors():
    '''For trained Support Vector Machines
    1. Calculate MAE
    2. Plot comparison with Vegas
    3. Plot Betting Simulation
    '''
    results = predict_support_vectors(df_scaled.drop(columns=['r_spread']))
    mae = mean_absolute_error(df_unscaled['r_spread'], results)
    logger.info(f"{SVM} MAE: {mae}")
    df_model = df_scaled.assign(pred = results)
    plot_error(df_model, SVM)
    plot_sim(results, SVM, 3)
    
def analyze_gradient_boosted_trees():
    ''' For trained Gradient Boosted Trees:
    1. Calculate MAE
    2. Plot comparison with Vegas
    3. Plot Betting Simulation
    '''
    results = predict_gbt(df_unscaled.drop(columns=['r_spread']))
    mae = mean_absolute_error(df_unscaled['r_spread'], results)
    logger.info(f"{GBT} MAE: {mae}")
    df_model = df_unscaled.assign(pred = results)
    plot_error(df_model, GBT)
    plot_sim(results, GBT, 2)

def analyze_nn():
    '''For trained Neural Netwrok
    1. Calculate MAE
    2. Plot comparison with Vegas
    3. Plot Betting Simulation
    '''
    results = predict_nn(df_scaled.drop(columns=['r_spread']))
    mae = mean_absolute_error(df_unscaled['r_spread'], results)
    logger.info(f"{NN} MAE: {mae}")
    df_model = df_scaled.assign(pred = results)
    plot_error(df_model, NN)
    plot_sim(results, NN, 3)

def plot_error(df_model: pd.DataFrame, model: str):
    df_vegas = pd.read_csv(str(vegas_path))
    r_spread_list = []
    def calc_vegas_error(r: pd.Series):
        r_spread = r.loc['tm_score'] - r.loc['opp_score']
        tm_spread = r.loc['tm_spread']
        if tm_spread <= 0.:
            r['vegas_error'] = max(0, abs(tm_spread) - r_spread)
        else:
            r['vegas_error'] = max(0, tm_spread + r_spread)
        r_spread_list.append(r_spread)
        return r

    df_base = df_vegas.apply(calc_vegas_error, axis=1)
    df_vegas_avg = df_base.loc[:, ['week','vegas_error']].set_index('week')
    vegas_avg_err = []
    weeks = range(1, 22)
    for week in weeks:
        vegas_avg_err.append(df_vegas_avg.loc[week, 'vegas_error'].mean())

    mae_vegas = np.mean(vegas_avg_err)

    #Model Error
    df_model['r_spread'] = r_spread_list
    def calc_error(r: pd.Series):
        r['error'] = abs(r['r_spread'] - r['pred'])
        return r

    df_model = df_model.apply(calc_error, axis=1)
    df_avg = df_model.loc[:, ['week','error']].set_index('week')
    avg_err = []
    weeks = range(1, 22)
    for week in weeks:
        avg_err.append(df_avg.loc[week, 'error'].mean())

    mae_model = mean_absolute_error(df_model['r_spread'], df_model['pred'])
    
    #Plotting
    plt.figure(figsize=(10,5))
    plt.plot(weeks, avg_err, marker='o', label=model+' Spread Error', color='b')
    plt.plot(weeks, vegas_avg_err, marker='x', label='Vegas Spread Error', color='g')

    # Plot the overall average line
    plt.axhline(y=mae_model, color='b', linestyle='--', label=model+f' MAE({mae_model:.2f})')
    plt.axhline(y=mae_vegas, color='g', linestyle='--', label=f'Vegas MAE({mae_vegas:.2f})')

    # Customize the x-axis to display every season
    plt.xticks(weeks)  # Rotating labels for better readability

    plt.xlabel('Week')
    plt.ylabel('Spread Error')
    plt.title(model+' vs Vegas Spread Error ')
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent label overlap
    path_name = model+"_rspread_error.png"
    fig_path = FIGURES_DIR / path_name
    if SAVE:
        plt.savefig(str(fig_path))
    plt.show()

def plot_sim(results, model: str, threshold: int):
    df_pred = pd.DataFrame({'r_spread': df_unscaled['r_spread'].values.tolist(), 'pred': results.flatten().tolist(), 'spread_line': df_unscaled['tm_spread'].values.tolist()})
    # Initialize variables for pnl and profit/loss
    current_pnl = 0
    wager_amount = 110  # Wager $110 per bet to win $100
    total_wagered = 0
    bets_won = 0
    total_games = df_pred.shape[0]

    # Initialize a list to store pnl after each bet
    pnl_progress = []

    # Loop through each row to simulate the bets
    for index, row in df_pred.iterrows():
        actual_r_spread = row['r_spread']
        prediction = row['pred']
        spread_line = row['spread_line']
        
        # Betting strategy: Bet on the team if pred > spread_line (predicting the team will cover the spread)
        if spread_line < 0 and (prediction - abs(spread_line)) > threshold:
            # If actual r_spread (team's win margin) covers the spread line, bet wins
            total_wagered+=wager_amount
            if actual_r_spread > abs(spread_line):
                current_pnl += 100  # Win $100
                bets_won+=1
            else:
                current_pnl -= wager_amount  # Lose $110 due to vigorish (110)
        elif spread_line > 0 and (abs(prediction) - spread_line) > threshold:
            # If actual r_spread (team's win margin) covers the spread line, bet wins
            total_wagered+=wager_amount
            if actual_r_spread > -spread_line:
                current_pnl += 100  # Win $100
                bets_won+=1
            else:
                current_pnl -= wager_amount  # Lose $110 due to vigorish (110)

        # Append the current pnl to the list for tracking progress
        pnl_progress.append(current_pnl)

    # Add the pnl progress to the DataFrame for visualization
    df_pred['pnl'] = pnl_progress
    squeezed_pnl = df_pred['pnl'].loc[df_pred['pnl'].shift() != df_pred['pnl']].reset_index(drop=True)

    logger.info(f'Total Games: {total_games}')
    logger.info(f'Total Bets: {round(total_wagered/110)}')
    logger.info(f"Total Wagered: {total_wagered}")
    logger.info(f"Total PnL: {current_pnl}")
    logger.info(f'Bets Won: {bets_won}')
    logger.info(f'Win Rate: {(bets_won/(total_wagered/110))*100}')
    logger.info(f'Return on Investment (ROI): {(current_pnl/(total_wagered))*100}')
    # Plot the pnl progression over time (as bets progress)
    plt.figure(figsize=(10, 6))
    plt.plot(squeezed_pnl, label='PNL Progress', color='b')
    plt.title(f'{model} Betting Simulation')
    plt.xlabel('Number of Bets')
    plt.ylabel('PnL ($)')
    plt.legend()
    plt.grid(True)
    plt.show()



if __name__ == "__main__":
    print("Not directly callable")
