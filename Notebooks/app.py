import streamlit as st

st.write("""
# Car Price Predictor
This app predicts **Used Cars Value**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

import pickle
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder

model = pickle.load(open('/users/alfon/Desktop/Master/TFM/Notebooks/RF_price_predicting_model.pkl','rb'))












