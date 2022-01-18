![CarGenius](CarGenius_header.png)

**Author**: [Alfonso Najera del Barrio](https://www.linkedin.com/in/alfonso-n%C3%A1jera-del-barrio-66b926124/)

Master Thesis in Data Science at [KSchool](https://kschool.com/) (2021-2022)

## Table of Contents

1. [Repository Structure](#Repository-structure)
2. [Introduction](#Introduction)
3. [Methodology](#Methodology)
4. [User Manual](#User-manual)
5. [Conclusions](#Conclusions)

<a name="Repository-structure"></a>
## Repository Structure

In the repository tree you will find the following **folders** :

* Brands : includes all car makes images .png used for the app visualiation

* Csv : includes all .csv files exported during the project

* Figs : figs saved during the project

* Files : files needed for the project, including the files downloaded to be able to use geopandas and the pickle file saved with the model

* Notebooks : included all jupyter notebooks, with the coding including all the cells executed and their results, used during the project for every process


You will find also the following **files**:

* CarGenius.py : app code

* CarGenius_header.png :app logo

* Procfile : set up files for heroku app

* requirements.txt : set up files for heroku app

* setup.sh :set up files for heroku app


<a name="Introduction"></a>
## Introduction
Car Genius is a used-car price predictor developed for my thesis in the Data Science Master from KSchool. The name is the result of combining the words Car and Genius , the last one as a synonym of Guru related to the aim of this project which is to build a machine learning  algorithm able to predict the price of any used car in Spain based on the current used cars market and applying the knowledge acquired during this master.

The project will cover all phases related to a Data Science project, involving the data acquisition and transformation, analysis, including the evaluation of the different machine learning models and will end with a frontend or visualization tool that will allow users to interact with the algorithm.

<a name="Methodology"></a>
## Methodology



<a name="User-manual"></a>
## User Manual

### Dependencies
For the correct execution of the project you will need the following dependencies installed on your anaconda environment:

- pandas 

- numpy

- matplotlib

- geopandas

- BeautifulSoup

- scikit-learn

- pickle

- streamlit

- heroku

### User Manual

The app has been designed to be very easy to use . When you access the webpage https://cargenius.herokuapp.com/ you will see:

Header: App logo and a short description about it

Sidebar: located on the left part of the page will allow users to specify their car options to allow the algorithm to make a prediction.

Specified Input parameters summary and the result of the prediction

Model predictions explanation using SHAP values

The user only needs to input the options of the car they want to make a prediction of by using the sidebar on the left of the screen “Specify Input Parameters''. The algorithm will run automatically and the predicted price for the car will appear under the “The predicted value for this cars is”.


<a name="Conclusions"></a>
## Conclusions

The purpose of this project was to create an accurate algorithm able to predict the used cars value for the Spanish Market and the Car Genius app serves the purpose it was developed for in a very friendly and easy to use format.

The project has been developed end to end, from the raw data extraction to an interactive app where the users can input the parameters to the algorithm to make predictions. 

During the project, more than 30,000 used cars ads were studied and although it is not a large number of ads for a Machine Learning project, the predictions obtained have been quite good, reaching an 89% accuracy using the Random Forest Model.



The most time consuming part of the project was the data source selection and the data cleaning. Once you decide what your work is going to be about you have to start thinking about how you will get the data for the study and find those sites with the relevant information and open to be scrapped. After that, data cleaning is one of the keys of the project, with quality data and data cleaned accordingly (formattings, outliers, re-groupings, checking inconsistencies…) the model will be able to provide better results.

For the future, it will be interesting to monitor these used car values on a temporary basis, in order to help buyers and sellers to choose the best moment for a sale or a purchase and help the users to see if a car model is being appreciated or depreciated.
