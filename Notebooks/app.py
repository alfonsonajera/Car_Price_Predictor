import streamlit as st
import pickle
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import bz2
import _pickle as cPickle

compressed_model = bz2.BZ2File("Notebooks/RF_price_predicting_model.pkl.pbz2", 'rb')
model = cPickle.load(compressed_model)



st.write("""
# Car Price Predictor
This app predicts **Used Cars Value**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

# Loads the Used Cars Dataset
cars_final = pd.read_csv('Csv/03.cars_final_def.csv')
cars_final = cars_final.drop(['Version', 'ZIP'], axis=1)
X= cars_final[cars_final.columns[:-1]]
y= cars_final[cars_final.columns[-1]]



