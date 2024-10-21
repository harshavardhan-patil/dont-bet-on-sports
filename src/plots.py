from pathlib import Path

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error as mae 
from src.config import FIGURES_DIR, INTERIM_DATA_DIR

vegas_path: Path = INTERIM_DATA_DIR / 'testset.csv'

def plot_error(df_model: pd.DataFrame, model: str):
    df_vegas = pd.read_csv(str(vegas_path))
    def calc_vegas_error(r: pd.Series):
        r_spread = r.loc['tm_score'] - r.loc['opp_score']
        tm_spread = r.loc['tm_spread']
        if tm_spread <= 0.:
            r['vegas_error'] = max(0, abs(tm_spread) - r_spread)
        else:
            r['vegas_error'] = max(0, tm_spread + r_spread)
        return r

    df_base = df_vegas.apply(calc_vegas_error, axis=1)
    df_vegas_avg = df_base.loc[:, ['week','vegas_error']].set_index('week')
    vegas_avg_err = []
    weeks = range(1, 22)
    for week in weeks:
        vegas_avg_err.append(df_vegas_avg.loc[week, 'vegas_error'].mean())

    mae_vegas = np.mean(vegas_avg_err)

    #Model Error
    def calc_error(r: pd.Series):
        r['error'] = abs(r['r_spread'] - r['pred'])
        return r

    df_model = df_model.apply(calc_error, axis=1)
    df_avg = df_model.loc[:, ['week','error']].set_index('week')
    avg_err = []
    weeks = range(1, 22)
    for week in weeks:
        avg_err.append(df_avg.loc[week, 'error'].mean())

    mae_model = mae(df_model['r_spread'], df_model['pred'])
    
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
    plt.savefig(str(fig_path))
    plt.show()

if __name__ == "__main__":
    print("Not directly callable")
