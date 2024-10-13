from pathlib import Path

import typer
from loguru import logger
from tqdm import tqdm
import numpy as np
import pandas as pd

from dbos.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR

def prepare_data(df : pd.DataFrame, is_train: bool) -> pd.DataFrame:
    #Removing certain columns based on logic mentioned in EDA notebook
    df = df.drop(columns=['won_toss_decision', 'won_toss_overtime', 'won_toss_overtime_decision', 'game_time', 'tm_pass_cmp_pct' , 'opp_pass_cmp_pct'])
    
    #filling domed/indoor stadiums weather conditions within a range
    df['temperature'] = df['temperature'].map(lambda t : t if not np.isnan(t) else float(np.random.randint(60., 76.)))
    df['wind_speed'] = df['wind_speed'].map(lambda t : t if not np.isnan(t) else float(np.random.randint(0., 3.)))
    df['humidity_pct'] = df['humidity_pct'].map(lambda t : t if t > 0.0 else float(np.random.randint(30., 51.)))

    #should save metadata
    #df_meta = df.loc[:,['event_date', 'season', 'tm_nano', 'tm_name', 'tm_market', 'tm_spread', 'opp_nano', 'opp_name', 'opp_market', 'opp_spread', 'status', 'opp_alt_market', 'tm_alt_market', 'total', 'tm_alt_alias', 'opp_alt_alias', 'attendance', 'duration']]
    df = df.drop(columns=['event_date', 'tm_nano', 'tm_name', 'tm_market', 'tm_spread', 'opp_nano', 'opp_name', 'opp_market', 'opp_spread', 'status', 'opp_alt_market', 'tm_alt_market', 'total', 'tm_alt_alias', 'opp_alt_alias', 'attendance', 'duration', 'surface_type'])
    
    def calc_r_spread(r: pd.Series):
        r['r_spread'] = r.loc['tm_score'] - r.loc['opp_score']
        return r
    df = df.apply(calc_r_spread, axis=1)
    df = df.drop(columns=['tm_score', 'opp_score'])

    df_stats = pd.DataFrame(columns=['alias', 'games','first_downs', 'fourth_down_att', 'fourth_down_conv',
       'fourth_down_conv_pct', 'fumbles', 'fumbles_lost',
       'net_pass_yds', 'pass_att', 'pass_cmp',
       'pass_int', 'pass_tds', 'pass_yds', 'passer_rating', 'penalties',
       'penalty_yds', 'rush_att', 'rush_tds', 'rush_yds', 'third_down_att',
       'third_down_conv', 'third_down_conv_pct', 'time_of_possession',
       'times_sacked', 'total_yds', 'turnovers', 'yds_sacked_for'])
    df_stats.set_index('alias', inplace=True)

    #ugly
    def calc_avg(team, total_games):
        avg = []
        for col in df_stats.columns.array[1:]:
            avg.append(df_stats.at[team, col] / total_games)
        return avg

    def get_team_features(team):
        if team not in df_stats.index or df_stats.at[team, 'games'] == 0:
            return pd.Series([0]*(len(df_stats.columns.array)-1), index=df_stats.columns.array[1:])
        
        total_games = df_stats.at[team, 'games']
        return pd.Series(calc_avg(team, total_games), index=df_stats.columns.array[1:])

    def remove_prefix(prefix):
        return lambda x: x[len(prefix):]

    def update_team_stats(r: pd.Series):
        team = r['alias']
        if team not in df_stats.index:
            df_stats.loc[team] = [0]*len(df_stats.columns.array)

        df_stats.loc[team, 'games'] += 1
        for col in df_stats.columns.array[1:]:
            df_stats.loc[team, col] += r[col]


    def calc_stats(r: pd.Series):
        tm_row = get_team_features(r.loc['tm_alias'])
        opp_row = get_team_features(r.loc['opp_alias'])
        update_team_stats(r.loc[df.filter(regex='^tm_').columns].rename(remove_prefix('tm_')))
        update_team_stats(r.loc[df.filter(regex='^opp_').columns].rename(remove_prefix('opp_')))
        tm_row = tm_row.add_prefix('tm_')
        opp_row = opp_row.add_prefix('opp_')
        for col in [ 'tm_first_downs',
       'tm_fourth_down_att', 'tm_fourth_down_conv', 'tm_fourth_down_conv_pct',
       'tm_fumbles', 'tm_fumbles_lost', 'tm_net_pass_yds', 'tm_pass_att',
       'tm_pass_cmp', 'tm_pass_int', 'tm_pass_tds', 'tm_pass_yds',
       'tm_passer_rating', 'tm_penalties', 'tm_penalty_yds', 'tm_rush_att',
       'tm_rush_tds', 'tm_rush_yds', 'tm_third_down_att', 'tm_third_down_conv',
       'tm_third_down_conv_pct', 'tm_time_of_possession', 'tm_times_sacked',
       'tm_total_yds', 'tm_turnovers', 'tm_yds_sacked_for', 'opp_first_downs',
       'opp_fourth_down_att', 'opp_fourth_down_conv',
       'opp_fourth_down_conv_pct', 'opp_fumbles', 'opp_fumbles_lost',
       'opp_net_pass_yds', 'opp_pass_att', 'opp_pass_cmp', 'opp_pass_int',
       'opp_pass_tds', 'opp_pass_yds', 'opp_passer_rating', 'opp_penalties',
       'opp_penalty_yds', 'opp_rush_att', 'opp_rush_tds', 'opp_rush_yds',
       'opp_third_down_att', 'opp_third_down_conv', 'opp_third_down_conv_pct',
       'opp_time_of_possession', 'opp_times_sacked', 'opp_total_yds',
       'opp_turnovers', 'opp_yds_sacked_for']:
            r[col] = tm_row.at[col] if col.startswith('tm_') else opp_row.at[col]
        
        return r

    df_processed = df.apply(calc_stats, axis=1)
    if is_train:
        stats = PROCESSED_DATA_DIR / "stats.csv"
        df_stats.to_csv(str(stats))
        logger.info("Created STATS")
    return pd.get_dummies(df_processed.drop(columns=['season']))
