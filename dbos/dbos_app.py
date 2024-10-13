import streamlit as st
import pandas as pd
import numpy as np
from dbos.config import DB_DATA_DIR
from dbos.modeling.predict import predict_random_forests
from dbos.features import prepare_data
#from loguru import logger

st.title("Don't bet on sports, kids!")

stats_path = DB_DATA_DIR / "stats.csv"
df_stats = pd.read_csv(str(stats_path))
#logger.info("Read File")

tm = st.selectbox(
    'Home Team',
     df_stats['alias'])

opp = st.selectbox(
    'Away Team',
     df_stats['alias'])

if st.button("Make Me MONEYYY!"):
    df_predict = prepare_data(tm, opp)
    r_spread = predict_random_forests(df_predict)[0]
    if r_spread > 0:
        st.subheader("_"+tm+" wins!"+"_", divider=True)
        st.subheader(tm+" spread line: -"+str(round(r_spread)))
    else:
        st.subheader("_"+opp+" wins!"+"_", divider=True)
        st.subheader(opp+" spread line: "+str(round(r_spread)))