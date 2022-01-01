import streamlit as st
import pickle
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import bz2
import _pickle as cPickle






st.write("""
# Car Price Predictor
This app predicts **Used Cars Value**!
""")
st.write('---')


compressed_model = bz2.BZ2File("Notebooks/RF_price_predicting_model.pkl.pbz2", 'rb')
model = cPickle.load(compressed_model)



