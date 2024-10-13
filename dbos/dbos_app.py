import streamlit as st
import pandas as pd
import numpy as np
from dbos.config import PROCESSED_DATA_DIR

st.title("Don't bet on sports, kids!")

stats_path = PROCESSED_DATA_DIR / "stats.csv"
df = pd.read_csv(str(stats_path))


tm = st.selectbox(
    'Which number do you like best?',
     df['alias'])

opp = st.selectbox(
    'Which number do you like best?',
     df['alias'])

