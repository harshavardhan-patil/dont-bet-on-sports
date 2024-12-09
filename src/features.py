from pathlib import Path

from loguru import logger
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

from src.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, DB_DATA_DIR, MODELS_DIR

stats_path = DB_DATA_DIR / "stats.csv"
input_struct_path = DB_DATA_DIR / "input_struct.csv"
scaled_train_path: Path = PROCESSED_DATA_DIR / "scaled" / "trainset.csv"
scaled_test_path: Path = PROCESSED_DATA_DIR / "scaled" /"testset.csv"
unscaled_train_path: Path = PROCESSED_DATA_DIR / "unscaled" / "trainset.csv"
unscaled_test_path: Path = PROCESSED_DATA_DIR / "unscaled" /"testset.csv"
robust_scaler_path = MODELS_DIR / "scalers" / "scaler_robust.gz"
scaler_path = MODELS_DIR / "scalers" / "scaler_y.gz"
min_max_scaler_path = MODELS_DIR / "scalers" / "scaler_minmax.gz"

#scaling for non-tree based models
def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    #one-hot encoded columns
    def filter_columns(column_names):
        return [col for col in column_names if not (col.startswith('tm_alias') 
                                                    or col.startswith('opp_alias') 
                                                    or col.startswith('won_toss') 
                                                    or col.startswith('tm_location') 
                                                    or col.startswith('opp_location') 
                                                    or col.startswith('roof_type')  
                                                    or col in ['week', 'wind_speed', 'r_spread'])]

    data_cols = filter_columns(df.columns.unique().to_list())

    #standardization with robust handling of outliers
    robust_scaler = RobustScaler()
    df.loc[:,data_cols] = robust_scaler.fit_transform(df.loc[:,data_cols])
    joblib.dump(robust_scaler, str(robust_scaler_path))

    #scale target variable and save scaler for getting true to life r_spread prediction
    scaler = StandardScaler()
    df.loc[:,['r_spread']] = scaler.fit_transform(df.loc[:,['r_spread']])
    joblib.dump(scaler, str(scaler_path))

    #normalizing right skewed wind_speed
    min_max_scaler = MinMaxScaler()
    df.loc[:,['wind_speed']] = min_max_scaler.fit_transform(df.loc[:,['wind_speed']])
    joblib.dump(min_max_scaler, str(min_max_scaler_path))

    
    return df
    
#for training and testing
def prepare_tt_data(is_train: bool):
    input_path = INTERIM_DATA_DIR / "trainset.csv" if is_train else INTERIM_DATA_DIR / "testset.csv"
    df = pd.read_csv(str(input_path), index_col='id')

    #Removing certain columns based on logic mentioned in EDA notebook
    df = df.drop(columns=['won_toss_decision', 'won_toss_overtime', 'won_toss_overtime_decision', 'game_time', 
                          'tm_pass_cmp_pct' , 'opp_pass_cmp_pct', 'event_date', 'tm_nano', 'tm_name', 'tm_market',
                             'opp_nano', 'opp_name', 'opp_market', 'status', 'opp_alt_market',
                              'tm_alt_market', 'tm_alt_alias', 'opp_alt_alias', 'attendance', 'duration', 'surface_type'])
    
    #filling domed/indoor stadiums weather conditions within a range
    df['temperature'] = df['temperature'].map(lambda t : t if not np.isnan(t) else float(np.random.randint(60., 76.)))
    df['wind_speed'] = df['wind_speed'].map(lambda t : t if not np.isnan(t) else float(np.random.randint(0., 3.)))
    df['humidity_pct'] = df['humidity_pct'].map(lambda t : t if t > 0.0 else float(np.random.randint(30., 51.)))

    def calc_r_spread(r: pd.Series):
        r['r_spread'] = r.loc['tm_score'] - r.loc['opp_score']
        return r
    df = df.apply(calc_r_spread, axis=1)
    df = df.drop(columns=['tm_score', 'opp_score'])

    #todo - weather averages and seasonal ewa
    df_stats = pd.DataFrame(columns=['alias', 'games','first_downs', 'fourth_down_att', 'fourth_down_conv',
       'fourth_down_conv_pct', 'fumbles', 'fumbles_lost',
       'net_pass_yds', 'pass_att', 'pass_cmp',
       'pass_int', 'pass_tds', 'pass_yds', 'passer_rating', 'penalties',
       'penalty_yds', 'rush_att', 'rush_tds', 'rush_yds', 'third_down_att',
       'third_down_conv', 'third_down_conv_pct', 'time_of_possession',
       'times_sacked', 'total_yds', 'turnovers', 'yds_sacked_for'])
    df_stats.set_index('alias', inplace=True)

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

    #populate running averages
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
    df_processed = pd.get_dummies(df_processed.drop(columns=['season']))

    if is_train:
        tm_names_path = DB_DATA_DIR / "tm_names.csv"
        df_tm_names = pd.read_csv(str(tm_names_path), index_col='alias')
        df_stats = df_stats.merge(df_tm_names, left_index = True, right_index=True,validate='one_to_one')
        df_stats.to_csv(str(stats_path))
        logger.info("Created stats")
        pd.DataFrame(0., index=[0], columns=df_processed.columns).to_csv(input_struct_path, index=False)
        logger.info("Created input struct")
        df_processed.to_csv(unscaled_train_path)
        scale_features(df_processed).to_csv(scaled_train_path)
        logger.info("Created trainset")
    else:
        df_processed.to_csv(unscaled_test_path)
        scale_features(df_processed).to_csv(scaled_test_path)
        logger.info("Created testset")
    
        
#for web app input
def prepare_data(tm, opp) -> pd.DataFrame:
    df = pd.read_csv(input_struct_path).drop(columns=['r_spread'])

    #set home and opp teams
    df['tm_alias_'+tm] = 1
    df['opp_alias_'+opp] = 1
    df['tm_location_H'] = 1
    df['opp_location_A'] = 1
    df['week'] = 16
    df['week_day_Sun'] = 1
    
    #filling domed/indoor stadiums weather conditions within a range todo-change to stadium wise weather
    df['temperature'] = float(np.random.randint(60., 76.))
    df['wind_speed'] = float(np.random.randint(0., 3.))
    df['humidity_pct'] = float(np.random.randint(30., 51.))
    df['roof_type_'+'Dome'] = 1

    #toss decision
    df['won_toss_'+np.random.choice([tm, opp])] = 1

    df_stats = pd.read_csv(stats_path).drop(columns=['name', 'market'])
    df_stats.set_index('alias', inplace=True)

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


    def calc_stats(r: pd.Series):
        tm_row = get_team_features(tm)
        opp_row = get_team_features(opp)
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

    #scaling should depend on the model being used - currently using NN
    #todo: fix testset scalers are used on train stats?
    def scale(df):
        def filter_columns(column_names):
            return [col for col in column_names if not (col.startswith('tm_alias') 
                                                        or col.startswith('opp_alias') 
                                                        or col.startswith('won_toss') 
                                                        or col.startswith('tm_location') 
                                                        or col.startswith('opp_location') 
                                                        or col.startswith('roof_type')  
                                                        or col in ['week', 'wind_speed'])]

        data_cols = filter_columns(df.columns.unique().to_list())
        robust_scaler = joblib.load(robust_scaler_path)
        df.loc[:,data_cols] = robust_scaler.transform(df.loc[:,data_cols])

        min_max_scaler = joblib.load(min_max_scaler_path)
        df.loc[:,['wind_speed']] = min_max_scaler.transform(df.loc[:,['wind_speed']])
        return df

    return scale(pd.get_dummies(df_processed))


if __name__ == "__main__":
    prepare_tt_data(True)