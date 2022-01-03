import streamlit as st
import pickle
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
import bz2
import _pickle as cPickle
import shap

compressed_model = bz2.BZ2File("Files/RF_price_predicting_model.pkl.pbz2", 'rb')
model = cPickle.load(compressed_model)



st.write("""
# VhehiCALC
This app predicts **Used Cars Values**!
""")
st.write('---')

# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

# Loads the Used Cars Dataset
cars_final = pd.read_csv('Csv/03.cars_final_def.csv')
cars_final = cars_final.drop(['Version', 'ZIP'], axis=1)
cars_final['Province'] = cars_final['Province'].fillna("Other")
X= cars_final[cars_final.columns[:-1]]
y= cars_final[cars_final.columns[-1]]


def user_input_features():
    
    
    BRAND = st.sidebar.selectbox('Brand', np.sort(cars_final.Brand.unique()), index = 8)
    MODEL = st.sidebar.selectbox('Model', np.sort(cars_final[cars_final.Brand == BRAND].Model.unique()), index=0)
        
    YEAR = st.sidebar.slider('Year', int(X.Year.min()), int(X.Year.max()), 2021)
    KMS = st.sidebar.number_input('Kms', 0, 1000000, 0, step = 1)
    HP = st.sidebar.slider('Power(Hp)', 0, 1000, 0)
    TRANSMISSION = st.sidebar.selectbox('Transmission', X.Gear_type.unique())
    FUEL = st.sidebar.selectbox('Fuel type', cars_final.Fuel_type.unique(), index=0)
    
    
    CONS = st.sidebar.slider('Fuel cons', int(X.Fuel_cons.min()), int(X.Fuel_cons.max()), int(X.Fuel_cons.mean()))
    DOORS = st.sidebar.slider('Doors', int(X.Doors.min()), int(X.Doors.max()), 5)
    COLOUR = st.sidebar.selectbox('Colour', np.sort(cars_final.Colour.unique()), index=12)
    TYPE = st.sidebar.selectbox('Type', cars_final.Type.unique(), index=0)
    
    
    
    PROVINCE = st.sidebar.selectbox('Province', np.sort(cars_final.Province.unique()), index=29)
    SELLER = st.sidebar.radio("Seller", ("Dealer", "Private"))
    
    
    
    
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

imag = "BMW-logo.png"

   

def main():
    
    
    
    # Main Panel

    # Print specified input parameters
    st.header('Specified Input parameters')
    st.write(df_frontend)
    st.image(imag, use_column_width=True)
    st.write('---')

    # Build Regression Model

    df = pd.concat([df_frontend, cars_final], axis=0).reset_index().drop('index', axis=1)

    for col in ['Brand','Gear_type', 'Fuel_type','Type','Seller']:
        df[col] = df[col].astype('category')

    df = pd.get_dummies(data=df,columns=['Gear_type','Fuel_type','Type','Seller'])

    encoder = TargetEncoder()


    cols_to_encode = ['Brand','Model', 'Colour', 'Province']
    cols_encoded = list(map(lambda c: c + '_encoded', cols_to_encode))

    df[cols_encoded] = encoder.fit_transform(df[cols_to_encode], df.Price_EUR)

    df.drop(['Brand','Model', 'Colour', 'Province', 'Price_EUR'], axis = 1, inplace = True)

    df_pred = df[:1]

    # Apply Model to Make Prediction

    prediction = pd.DataFrame(model.predict(df_pred))
    prediction.columns = ['Price_EUR']
    prediction['Price_EUR']= prediction['Price_EUR'].map('€{:,.0f}'.format)





  
  

    st.header('The predicted value for this cars is:')
    st.write(prediction)
    st.write('---')
    
    

if __name__ == "__main__":
    main()
