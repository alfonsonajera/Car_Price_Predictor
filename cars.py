{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "9bb6dbf0",
   "metadata": {},
   "source": [
    "# 07. Model Deployment"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0026a0b0",
   "metadata": {},
   "source": [
    "!pip install ipykernel>=5.1.2\n",
    "!pip install pydeck\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a95bd822",
   "metadata": {},
   "source": [
    "conda install -c conda-forge shap"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7dbc0bed",
   "metadata": {},
   "source": [
    "!pip install streamlit"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "354e7759",
   "metadata": {},
   "source": [
    "pip install -U scikit-learn"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "2ef21a58",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import shap\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import datasets\n",
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0afe932f",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ccd00731",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[0m\r\n",
      "\u001b[34m\u001b[1m  Welcome to Streamlit. Check out our demo in your browser.\u001b[0m\r\n",
      "\u001b[0m\r\n",
      "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\r\n",
      "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://192.168.86.25:8501\u001b[0m\r\n",
      "\u001b[0m\r\n",
      "  Ready to create your own Python apps super quickly?\u001b[0m\r\n",
      "  Head over to \u001b[0m\u001b[1mhttps://docs.streamlit.io\u001b[0m\r\n",
      "\u001b[0m\r\n",
      "  May you create awesome apps!\u001b[0m\r\n",
      "\u001b[0m\r\n",
      "\u001b[0m\r\n"
     ]
    }
   ],
   "source": [
    "\n",
    "!streamlit hello\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5d3bfbcc",
   "metadata": {},
   "source": [
    "pip install watchdog"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8ef8b4bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = pickle.load(open('RF_price_predicting_model.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "48b0bdb6",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2022-01-01 17:48:11.396 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run /opt/anaconda3/envs/geo_env/lib/python3.9/site-packages/ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "st.title('VehiCALC')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c76545ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "st.write(\"\"\"\n",
    "# Boston House Price Prediction App\n",
    "This app predicts the **Boston House Price**!\n",
    "\"\"\")\n",
    "st.write('---')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "3063e63c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Loads the Used Cars Dataset\n",
    "cars_final = pd.read_csv('/users/alfon/Desktop/Master/TFM/CSV/03.cars_final_def.csv')\n",
    "cars_final = cars_final.drop(['Version', 'ZIP'], axis=1)\n",
    "X= cars_final[cars_final.columns[:-1]]\n",
    "y= cars_final[cars_final.columns[-1]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "ec4b6c76",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=1, _provided_cursor=None, _parent=DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None), _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Sidebar\n",
    "# Header of Specify Input Parameters\n",
    "st.sidebar.header('Specify Input Parameters')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12a68af6",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7265f65f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def user_input_features():\n",
    "    YEAR = st.sidebar.slider('Year', int(X.Year.min()), int(X.Year.max()), int(X.Year.mean()))\n",
    "    KMS = st.sidebar.slider('Kms', int(X.Kms.min()), int(X.Kms.max()), int(X.Kms.mean()))\n",
    "    HP = st.sidebar.slider('Hp', int(X.Hp.min ()), int(X.Hp.max()), int(X.Hp.mean()))\n",
    "    CONS = st.sidebar.slider('Fuel_cons', int(X.Fuel_cons.min()), int(X.Fuel_cons.max()), int(X.Fuel_cons.mean()))\n",
    "    DOORS = st.sidebar.slider('Doors', int(X.Doors.min()), int(X.Doors.max()), int(X.Doors.mean()))\n",
    "    TRANSMISSION = st.sidebar.selectbox('Gear_type', X.Gear_type.unique())\n",
    "    SELLER = st.sidebar.selectbox('Seller', X.Seller.unique())\n",
    "    \n",
    "    \n",
    "\n",
    "    BRAND = st.sidebar.selectbox('Brand', np.sort(cars_final.Brand.unique()), index=0, help='Choose car brand')\n",
    "    MODEL = st.sidebar.selectbox('Model', np.sort(cars_final[cars_final.Brand == BRAND].Model.unique()), index=0, help='Models available for the selected brand')\n",
    "    TYPE = st.sidebar.selectbox('Type', cars_final.Type.unique(), index=0)\n",
    "    PROVINCE = st.sidebar.selectbox('Province', cars_final.Province.unique(), index=0)\n",
    "    COLOUR = st.sidebar.selectbox('Colour', cars_final.Colour.unique(), index=0)\n",
    "    FUEL = st.sidebar.selectbox('Fuel_type', cars_final.Fuel_type.unique(), index=0)\n",
    "        \n",
    "\n",
    "        \n",
    "    \n",
    "    \n",
    "    \n",
    "    data = {'Brand': BRAND,\n",
    "            'Model': MODEL,\n",
    "            'Type': TYPE,\n",
    "            'Year': YEAR,\n",
    "            'Kms': KMS,\n",
    "            'Hp': HP,\n",
    "            'Gear_type': TRANSMISSION,\n",
    "            'Fuel_type': FUEL,\n",
    "            'Fuel_cons': CONS,\n",
    "            'Doors': DOORS,\n",
    "            'Colour': COLOUR,\n",
    "            'Province': PROVINCE,\n",
    "            'Seller': SELLER}\n",
    "    \n",
    "    features = pd.DataFrame(data, index=[0])\n",
    "    return features\n",
    "\n",
    "df_frontend = user_input_features()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5ced3ffb",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main Panel\n",
    "\n",
    "# Print specified input parameters\n",
    "st.header('Specified Input parameters')\n",
    "st.write(df_frontend)\n",
    "st.write('---')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "cf798e46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "65d9f613",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Brand</th>\n",
       "      <th>Model</th>\n",
       "      <th>Type</th>\n",
       "      <th>Year</th>\n",
       "      <th>Kms</th>\n",
       "      <th>Hp</th>\n",
       "      <th>Gear_type</th>\n",
       "      <th>Fuel_type</th>\n",
       "      <th>Fuel_cons</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Colour</th>\n",
       "      <th>Province</th>\n",
       "      <th>Seller</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Abarth</td>\n",
       "      <td>124 Spider</td>\n",
       "      <td>small</td>\n",
       "      <td>2013</td>\n",
       "      <td>106781</td>\n",
       "      <td>172</td>\n",
       "      <td>Manual</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>6</td>\n",
       "      <td>4</td>\n",
       "      <td>Beige</td>\n",
       "      <td>Barcelona</td>\n",
       "      <td>Dealer</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Brand       Model   Type  Year     Kms   Hp Gear_type Fuel_type  \\\n",
       "0  Abarth  124 Spider  small  2013  106781  172    Manual    Diesel   \n",
       "\n",
       "   Fuel_cons  Doors Colour   Province  Seller  \n",
       "0          6      4  Beige  Barcelona  Dealer  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_frontend.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "bd5651ac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Brand</th>\n",
       "      <th>Model</th>\n",
       "      <th>Type</th>\n",
       "      <th>Year</th>\n",
       "      <th>Kms</th>\n",
       "      <th>Hp</th>\n",
       "      <th>Gear_type</th>\n",
       "      <th>Fuel_type</th>\n",
       "      <th>Fuel_cons</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Colour</th>\n",
       "      <th>Province</th>\n",
       "      <th>Seller</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>15130</th>\n",
       "      <td>Mercedes-Benz</td>\n",
       "      <td>B Class</td>\n",
       "      <td>minivan</td>\n",
       "      <td>2016</td>\n",
       "      <td>125276</td>\n",
       "      <td>109</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>4.0</td>\n",
       "      <td>5</td>\n",
       "      <td>Silver</td>\n",
       "      <td>Castellón</td>\n",
       "      <td>Dealer</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Brand    Model     Type  Year     Kms   Hp  Gear_type  \\\n",
       "15130  Mercedes-Benz  B Class  minivan  2016  125276  109  Automatic   \n",
       "\n",
       "      Fuel_type  Fuel_cons  Doors  Colour   Province  Seller  \n",
       "15130    Diesel        4.0      5  Silver  Castellón  Dealer  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X.sample(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2ddb9d30",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([df_frontend, X], axis=0).reset_index().drop('index', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a92e2dd0",
   "metadata": {},
   "outputs": [],
   "source": [
    "for col in ['Brand','Gear_type', 'Fuel_type','Type','Seller']:\n",
    "    df[col] = df[col].astype('category')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "41aa30cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.get_dummies(data=df,columns=['Gear_type','Fuel_type','Type','Seller'])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80387450",
   "metadata": {},
   "source": [
    "pip install category_encoders"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "ace370f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from category_encoders import TargetEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "a8f1d1c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = TargetEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "7a129b17",
   "metadata": {},
   "outputs": [],
   "source": [
    "cols_to_encode = ['Brand','Model', 'Colour', 'Province']\n",
    "cols_encoded = list(map(lambda c: c + '_encoded', cols_to_encode))\n",
    "\n",
    "df[cols_encoded] = encoder.fit_transform(df[cols_to_encode], df.Year)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "60619b4e",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop(['Brand','Model', 'Colour', 'Province'], axis = 1, inplace = True)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d0dd28e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_pred = df[:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "3cad908f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Year</th>\n",
       "      <th>Kms</th>\n",
       "      <th>Hp</th>\n",
       "      <th>Fuel_cons</th>\n",
       "      <th>Doors</th>\n",
       "      <th>Gear_type_Automatic</th>\n",
       "      <th>Gear_type_Manual</th>\n",
       "      <th>Fuel_type_CNG</th>\n",
       "      <th>Fuel_type_Diesel</th>\n",
       "      <th>Fuel_type_Electric</th>\n",
       "      <th>...</th>\n",
       "      <th>Type_sedan</th>\n",
       "      <th>Type_small</th>\n",
       "      <th>Type_suv</th>\n",
       "      <th>Type_van</th>\n",
       "      <th>Seller_Dealer</th>\n",
       "      <th>Seller_Private</th>\n",
       "      <th>Brand_encoded</th>\n",
       "      <th>Model_encoded</th>\n",
       "      <th>Colour_encoded</th>\n",
       "      <th>Province_encoded</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2013</td>\n",
       "      <td>106781</td>\n",
       "      <td>172</td>\n",
       "      <td>6.0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2016.979695</td>\n",
       "      <td>2017.052632</td>\n",
       "      <td>2012.451389</td>\n",
       "      <td>2012.972741</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 28 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   Year     Kms   Hp  Fuel_cons  Doors  Gear_type_Automatic  Gear_type_Manual  \\\n",
       "0  2013  106781  172        6.0      4                    0                 1   \n",
       "\n",
       "   Fuel_type_CNG  Fuel_type_Diesel  Fuel_type_Electric  ...  Type_sedan  \\\n",
       "0              0                 1                   0  ...           0   \n",
       "\n",
       "   Type_small  Type_suv  Type_van  Seller_Dealer  Seller_Private  \\\n",
       "0           1         0         0              1               0   \n",
       "\n",
       "   Brand_encoded  Model_encoded  Colour_encoded  Province_encoded  \n",
       "0    2016.979695    2017.052632     2012.451389       2012.972741  \n",
       "\n",
       "[1 rows x 28 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3764f58e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "955e213c",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fab470e8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2109103e",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eef0842d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fb32cd9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6233028",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "783a2043",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Apply Model to Make Prediction\n",
    "\n",
    "prediction = model.predict(df_pred)\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "e69537b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.header('Prediction of MEDV')\n",
    "st.write(prediction)\n",
    "st.write('---')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "270466e1",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e3c587a4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e0b4466d",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "504f1f8b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1b45b45",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f8ce7ed7",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "934b82fd",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "241ecd23",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "23590bb0",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
