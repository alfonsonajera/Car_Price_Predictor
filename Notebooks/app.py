import streamlit as st
import pickle
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

st.write("""
# Car Price Predictor
This app predicts **Used Cars Value**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')



model = pickle.load(open('Notebooks/RF_price_predicting_model.pkl','rb'))
