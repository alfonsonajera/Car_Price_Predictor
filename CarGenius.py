import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
import bz2
import _pickle as cPickle
import shap

compressed_model = bz2.BZ2File("Files/RF_price_predicting_model.pkl.pbz2", 'rb')
model = cPickle.load(compressed_model)

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #000000;">
  <a class="navbar-brand" href="https://youtube.com/dataprofessor" target="_blank">Car Genius</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item">
        <a class="nav-link" href="https://youtube.com/dataprofessor" target="_blank">YouTube</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://twitter.com/thedataprof" target="_blank">Twitter</a>
      </li>
    </ul>
  </div>
</nav>
""", unsafe_allow_html=True)
# Title logo
st.image("CarGenius_header.png")
st.write(""" 
**Car Genius** will help you to predict Used Cars Values for the Spanish Market using a Machine Learning Algorithm updated up to 2021 to arrive at a fair value in just a few clicks!
""")         
st.write('---')


# Sidebar
# Header of Specify Input Parameters

side_bar = """
  <style>
    /* The whole sidebar */
    .css-1lcbmhc.e1fqkh3o0{
      margin-top: 3.8rem;
    }
     
     /* The display arrow */
    .css-sg054d.e1fqkh3o3 {
      margin-top: 5rem;
      }
  </style> 
  """
st.markdown(side_bar, unsafe_allow_html=True)
st.sidebar.header('Specify Input Parameters')

    
# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')


# Loads the Used Cars Dataset
cars_final = pd.read_csv('Csv/03.cars_final_def.csv')
cars_final = cars_final.drop(['Version', 'ZIP'], axis=1)
cars_final['Province'] = cars_final['Province'].fillna("Other")
X= cars_final[cars_final.columns[:-1]]
y= cars_final[cars_final.columns[-1]]


def alfaromeo():
    st.image("Brands/alfa-romeo-logo.png", use_column_width=False)
def audi():
    st.image("Brands/audi-logo.png", use_column_width=False)
def bmw():
    st.image("Brands/bmw-logo.png", use_column_width=False)
def citroen():
    st.image("Brands/citroen-logo.png", use_column_width=False)
def ferrari():
    st.image("Brands/ferrari-logo.png", use_column_width=False)
def fiat():
    st.image("Brands/fiat-logo.png", use_column_width=False)
def ford():
    st.image("Brands/ford-logo.png", use_column_width=False)
def honda():
    st.image("Brands/honda-logo.png", use_column_width=False)
def hyundai():
    st.image("Brands/hyundai-logo.png", use_column_width=False)
def jaguar():
    st.image("Brands/jaguar-logo.png", use_column_width=False)
def jeep():
    st.image("Brands/jeep-logo.png", use_column_width=False)
def kia():
    st.image("Brands/kia-logo.png", use_column_width=False)
def landrover():
    st.image("Brands/land-rover-logo.png", use_column_width=False)
def mazda():
    st.image("Brands/mazda-logo.png", use_column_width=False)
def mercedes():
    st.image("Brands/mercedes-benz-logo.png", use_column_width=False)
def mini():
    st.image("Brands/mini-logo.png", use_column_width=False)
def opel():
    st.image("Brands/opel-logo.png", use_column_width=False)
def peugeot():
    st.image("Brands/peugeot-logo.png", use_column_width=False)
def porsche():
    st.image("Brands/porsche-logo.png", use_column_width=False)
def renault():
    st.image("Brands/renault-logo.png", use_column_width=False)
def seat():
    st.image("Brands/seat-logo.png", use_column_width=False)
def skoda():
    st.image("Brands/skoda-logo.png", use_column_width=False)
def tesla():
    st.image("Brands/tesla-logo.png", use_column_width=False)
def toyota():
    st.image("Brands/toyota-logo.png", use_column_width=False)
def volkswagen():
    st.image("Brands/volkswagen-logo.png", use_column_width=False)
def volvo():
    st.image("Brands/volvo-logo.png", use_column_width=False)
def unknown():
    st.image("Brands/unknown-logo.png", use_column_width=False)

    

    
    
    
def user_input_features():
    
    
    BRAND = st.sidebar.selectbox('Brand', np.sort(cars_final.Brand.unique()), index = 8)
    
    if BRAND == "Alfa-Romeo":
        alfaromeo()
    elif BRAND == "Audi":
        audi()
    elif BRAND == "BMW":
        bmw()
    elif BRAND == "Citroen":
        citroen()
    elif BRAND == "Ferrari":
        ferrari()
    elif BRAND == "Fiat":
        fiat()
    elif BRAND == "Ford":
        ford()
    elif BRAND == "Honda":
        honda()
    elif BRAND == "Hyundai":
        hyundai()
    elif BRAND == "Jaguar":
        jaguar()
    elif BRAND == "Jeep":
        jeep()
    elif BRAND == "KIA":
        kia() 
    elif BRAND == "Land-Rover":
        landrover()
    elif BRAND == "Mazda":
        mazda()
    elif BRAND == "Mercedes-Benz":
        mercedes()
    elif BRAND == "MINI":
        mini()
    elif BRAND == "Opel":
        opel()
    elif BRAND == "Peugeot":
        peugeot()
    elif BRAND == "Porsche":
        porsche()
    elif BRAND == "Renault":
        renault()
    elif BRAND == "SEAT":
        seat()
    elif BRAND == "Skoda":
        skoda()
    elif BRAND == "Tesla":
        tesla()
    elif BRAND == "Toyota":
        toyota()
    elif BRAND == "Volkswagen":
        volkswagen() 
    elif BRAND == "Volvo":
        volvo()
    else:
        unknown()
        
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


    
def main():
    
    # Print specified input parameters
 
    st.write(df_frontend)
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
    
    # Explaining the model's predictions using SHAP values
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    st.header('Features impact to the model')
    st.image("Figs/07_Shap_summary_plot.png")
    st.image("Figs/07_Shap_summary_plot_bar.png")
    


    st.write('---')
    st.write("Designed by Alfonso Najera del Barrio")    
  
        


if __name__ == "__main__":
    main()
