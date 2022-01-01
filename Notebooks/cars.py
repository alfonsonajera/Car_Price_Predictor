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
   "id": "8cce46f9",
   "metadata": {},
   "source": [
    "pip install watchdog"
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
   "execution_count": 33,
   "id": "49687477",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "UsageError: %%writefile is a cell magic, but the cell body is empty.\n"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "2ef21a58",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Writing app.py\n"
     ]
    }
   ],
   "source": [
    "%%writefile app.py\n",
    "import streamlit as st\n",
    "import pickle\n",
    "import shap\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn import datasets\n",
    "from sklearn.ensemble import RandomForestRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "8ef8b4bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = pickle.load(open('RF_price_predicting_model.pkl','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "48b0bdb6",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 36,
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
   "execution_count": 37,
   "id": "c76545ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "st.write(\"\"\"\n",
    "# Boston House Price Prediction App\n",
    "This app predicts the **Boston House Price**!\n",
    "\"\"\")\n",
    "st.write('---')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
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
   "execution_count": 15,
   "id": "ec4b6c76",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator(_root_container=1, _provided_cursor=None, _parent=DeltaGenerator(_root_container=0, _provided_cursor=None, _parent=None, _block_type=None, _form_data=None), _block_type=None, _form_data=None)"
      ]
     },
     "execution_count": 15,
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
   "execution_count": 16,
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
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "9efa072a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "    \n",
    "    # Main Panel\n",
    "\n",
    "    # Print specified input parameters\n",
    "    st.header('Specified Input parameters')\n",
    "    st.write(df_frontend)\n",
    "    st.write('---')\n",
    "\n",
    "    # Build Regression Model\n",
    "\n",
    "    df = pd.concat([df_frontend, X], axis=0).reset_index().drop('index', axis=1)\n",
    "\n",
    "    for col in ['Brand','Gear_type', 'Fuel_type','Type','Seller']:\n",
    "        df[col] = df[col].astype('category')\n",
    "\n",
    "    df = pd.get_dummies(data=df,columns=['Gear_type','Fuel_type','Type','Seller'])\n",
    "\n",
    "    encoder = TargetEncoder()\n",
    "\n",
    "\n",
    "    cols_to_encode = ['Brand','Model', 'Colour', 'Province']\n",
    "    cols_encoded = list(map(lambda c: c + '_encoded', cols_to_encode))\n",
    "\n",
    "    df[cols_encoded] = encoder.fit_transform(df[cols_to_encode], df.Year)\n",
    "\n",
    "    df.drop(['Brand','Model', 'Colour', 'Province'], axis = 1, inplace = True)\n",
    "\n",
    "    df_pred = df[:1]\n",
    "\n",
    "    # Apply Model to Make Prediction\n",
    "\n",
    "    prediction = model.predict(df_pred)\n",
    "\n",
    "    st.header('Prediction of MEDV')\n",
    "    st.write(prediction)\n",
    "    st.write('---')\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
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
   "execution_count": null,
   "id": "cf798e46",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build Regression Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
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
     "execution_count": 18,
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
   "execution_count": 19,
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
       "      <th>7463</th>\n",
       "      <td>SsangYong</td>\n",
       "      <td>Actyon</td>\n",
       "      <td>suv</td>\n",
       "      <td>2007</td>\n",
       "      <td>138667</td>\n",
       "      <td>141</td>\n",
       "      <td>Automatic</td>\n",
       "      <td>Diesel</td>\n",
       "      <td>8.5</td>\n",
       "      <td>5</td>\n",
       "      <td>Blue</td>\n",
       "      <td>Málaga</td>\n",
       "      <td>Dealer</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Brand   Model Type  Year     Kms   Hp  Gear_type Fuel_type  \\\n",
       "7463  SsangYong  Actyon  suv  2007  138667  141  Automatic    Diesel   \n",
       "\n",
       "      Fuel_cons  Doors Colour Province  Seller  \n",
       "7463        8.5      5   Blue   Málaga  Dealer  "
      ]
     },
     "execution_count": 19,
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
   "execution_count": 20,
   "id": "2ddb9d30",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.concat([df_frontend, X], axis=0).reset_index().drop('index', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
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
   "execution_count": 22,
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
   "execution_count": 23,
   "id": "ace370f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from category_encoders import TargetEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "a8f1d1c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "encoder = TargetEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
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
   "execution_count": 26,
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
   "execution_count": 27,
   "id": "d0dd28e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_pred = df[:1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
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
     "execution_count": 28,
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
   "execution_count": 29,
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
   "execution_count": 30,
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
   "source": [
    "if __name__ == \"__main__\":\n",
    "  main()"
   ]
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
