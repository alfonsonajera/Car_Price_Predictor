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
