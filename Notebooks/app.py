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



model = pickle.load(open('/Users/alfon/Desktop/Master/TFM/Notebooks/RF_price_predicting_model.pkl','rb'))

# Loads the Used Cars Dataset
cars_final = pd.read_csv('/users/alfon/Desktop/Master/TFM/CSV/03.cars_final_def.csv')
cars_final = cars_final.drop(['Version', 'ZIP'], axis=1)
X= cars_final[cars_final.columns[:-1]]
y= cars_final[cars_final.columns[-1]]

def user_input_features():
    YEAR = st.sidebar.slider('Year', int(X.Year.min()), int(X.Year.max()), int(X.Year.mean()))
    KMS = st.sidebar.slider('Kms', int(X.Kms.min()), int(X.Kms.max()), int(X.Kms.mean()))
    HP = st.sidebar.slider('Hp', int(X.Hp.min ()), int(X.Hp.max()), int(X.Hp.mean()))
    CONS = st.sidebar.slider('Fuel_cons', int(X.Fuel_cons.min()), int(X.Fuel_cons.max()), int(X.Fuel_cons.mean()))
    DOORS = st.sidebar.slider('Doors', int(X.Doors.min()), int(X.Doors.max()), int(X.Doors.mean()))
    TRANSMISSION = st.sidebar.selectbox('Gear_type', X.Gear_type.unique())
    SELLER = st.sidebar.selectbox('Seller', X.Seller.unique())
    
    

    BRAND = st.sidebar.selectbox('Brand', np.sort(cars_final.Brand.unique()), index=0, help='Choose car brand')
    MODEL = st.sidebar.selectbox('Model', np.sort(cars_final[cars_final.Brand == BRAND].Model.unique()), index=0, help='Models available for the selected brand')
    TYPE = st.sidebar.selectbox('Type', cars_final.Type.unique(), index=0)
    PROVINCE = st.sidebar.selectbox('Province', cars_final.Province.unique(), index=0)
    COLOUR = st.sidebar.selectbox('Colour', cars_final.Colour.unique(), index=0)
    FUEL = st.sidebar.selectbox('Fuel_type', cars_final.Fuel_type.unique(), index=0)
        

        
    
    
    
    data = {'Brand': BRAND,
            'Model': MODEL,
            'Type': TYPE,
            'Year': YEAR,
            'Kms': KMS,
            'Hp': HP,
            'Gear_type': TRANSMISSION,
            'Fuel_type': FUEL,
            'Fuel_cons': CONS,
            'Doors': DOORS,
            'Colour': COLOUR,
            'Province': PROVINCE,
            'Seller': SELLER}
    
    features = pd.DataFrame(data, index=[0])
    return features

df_frontend = user_input_features()








