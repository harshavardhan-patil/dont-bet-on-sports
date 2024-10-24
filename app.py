import streamlit as st
import pandas as pd
import numpy as np
from src.config import DB_DATA_DIR
from src.modeling.predict import predict_nn
from src.features import prepare_data
import base64
from loguru import logger

#converting static image and setting as website background
def set_jpg_as_page_bg(bg):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/jpg;base64,{base64.b64encode(open(bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )


def get_name(alias: str):
    market = df_stats.at[alias, 'market']
    name = df_stats.at[alias, 'name']
    return market + ' ' + name

set_jpg_as_page_bg('static/bg.jpg')

st.title("Don't bet on sports, kids!")

col1, col2, col3 = st.columns([6, 4, 6])

stats_path = DB_DATA_DIR / "stats.csv"
df_stats = pd.read_csv(str(stats_path), index_col='alias')
logger.info("Read File")

tm = col1.selectbox(
    'Home Team',
     df_stats.index,
     format_func=get_name)

opp = col3.selectbox(
    'Away Team',
     df_stats.index,
     format_func=get_name)

with col2:
    st.markdown("[![Img Attr](./app/static/vs.png)](https://pngtree.com/freepng/vector-illustration-of-versus-battle-vs-icon-black-orange-color_5740367.html')")

if st.button("Make Me MONEYYY!", icon= "🤑"):
    tm_name = get_name(tm)
    opp_name = get_name(opp)
    df_predict = prepare_data(tm, opp)
    r_spread = predict_nn(df_predict)[0][0]
    if r_spread > 0:
        st.header("_"+tm_name+" win!"+"_")
        st.subheader(tm_name+" Spread Line: -"+str(round(r_spread)))
    else:
        st.header("_"+opp_name+" win!"+"_")
        st.subheader(opp_name+" Spread Line: "+str(round(r_spread)))