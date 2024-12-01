## Project: Weather Forecasting Model using Neural Networks (Project 1)
## Author: Austyn Griego
## Course: CS3210 - Machine Learning
## Date: November 23rd, 2024
## Project Title: Not So Accurate Weather App

## OPTION A -- Web Application using trained model
Option A: Build a Web App for a Machine Learning Model
Train a machine learning model on a dataset of your choice and create a web-based application to allow users to interact with it. You may use Flask or a similar web framework (here is a brief tutorial on using FlaskLinks to an external site.). This short lessonLinks to an external site. in a Microsoft online course on ML also covers how to "pickle" your model so that it can be used by your app. 

Proposal Requirements
Model Overview:
Briefly describe the machine learning model you plan to use (this can be a very basic type of model, given that you will be creating an app for it) -- NEURAL NETWORK
Include a justification for choosing this model, and point out why another type of model might not be a good choice
Dataset -- WEATHER PATTERNS, WEATHER COMPLEXITY
Describe the dataset you will use to train your model, including the source and the features. -- NOAA DATASET
App Functionality
Describe how users will interact with the model through the web app -- USER ENTERS ZIPCODE THEY WANT A FORECAST FOR
Outline any additional features (e.g., user input, visualizations, feedback) -- RETURN A FORECAST OF HIGH LOW RAIN/SNOW/AVG WIND
Tools and Technologies
Specify which frameworks or tools (e.g., Flask, HTML, CSS, etc.) you will use to build the app. -- FLASK, HTML, COPILOT, TENSFLOW

## Overview
The Not So Accurate Weather App is a web application that provides weather forecasts based on historical weather data from NOAA. The app uses Neural Network machine learning models to predict various weather parameters such as maximum temperature (TMAX), minimum temperature (TMIN), precipitation (PRCP), snowfall (SNOW), and average wind speed (AWND). The app is built using Flask and provides a user-friendly interface for entering a zipcode and viewing the (not-so-accurate) forecast.

## Features
- Predicts weather parameters: TMAX, TMIN, PRCP, SNOW, and AWND using pretrained Neural Network machine learning models.
- Displays icons and a card featuring the forecasted weather conditions.
- Styling using html and CSS, i have zero experience in these languages, however, i was able to make site look better to the user, with added colors, and background fill-in.
- Based on user inputed zipcode - app gets user longitude and lattitude to find nearby stations to gather and aggregate historical data from past 30 days and make new predictions.
- Error Handling, and debugging statements throughout code, including user informative error messages on the web application.

## Future Features 
These are the features I wanted or would still like to add outside of the project scope:
- Historical data visuals (graphs, charts, weather maps, etc.)\
- 7 Day Forecast model/visual
- More specified/useful predictions on data

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/not-so-accurate-weather-app.git
   cd not-so-accurate-weather-app
Expected Model Output: (Does not include whole terminal)

Requirements.txt has all imports required to run on venv

## Model Output
Predictions:
        TMAX      TMIN      PRCP      SNOW      AWND
0   3.518409  3.300333  0.077214  0.291762  2.072151
1   3.123149  2.364408  0.010687  0.022038  1.670326
2   3.597796  2.872427 -0.000140 -0.038106  2.421059
3   3.605472  3.144213  0.006228 -0.048102  2.119875
4   3.549781  2.926470  0.021589 -0.031075  2.506429
5   4.061012  3.577000  0.009678  0.274610  1.624122
6   3.556608  1.672168  0.000084  0.177821  1.422769
7   3.706184  2.971543 -0.001977 -0.036174  2.642668
8   3.650371  3.086612  0.010154 -0.041837  2.249779
9   3.611189  2.789905 -0.002981  0.057886  2.086437
10  3.756449  3.111994 -0.003408 -0.058065  2.335937
11  3.292114  2.266659 -0.010398  0.014362  2.029967

TMAX - Cross-validated MAE scores: [0.4149515628814697, 0.4953131377696991, 0.8685706853866577, 0.3500292897224426, 0.6028550267219543]
TMAX - Mean MAE: 0.5463439404964447, Std MAE: 0.18189619007538338

TMIN - Cross-validated MAE scores: [0.1782221645116806, 0.32035788893699646, 0.7978622317314148, 0.3646543622016907, 0.5387335419654846]
TMIN - Mean MAE: 0.43996603786945343, Std MAE: 0.21275926229873193

PRCP - Cross-validated MAE scores: [0.019182531163096428, 0.009041990153491497, 0.051679596304893494, 0.010339982807636261, 0.021017808467149734]
PRCP - Mean MAE: 0.022252381779253483, Std MAE: 0.015448810989661999

SNOW - Cross-validated MAE scores: [0.04460147023200989, 0.0419270284473896, 0.10812094062566757, 0.0446554571390152, 0.0767999142408371]
SNOW - Mean MAE: 0.06322096213698387, Std MAE: 0.025865773041333324

AWND - Cross-validated MAE scores: [0.2537437975406647, 0.24622957408428192, 0.5693408250808716, 0.21897409856319427, 0.40045130252838135]
AWND - Mean MAE: 0.33774791955947875, Std MAE: 0.13198977896209416

TMAX - Loss: 0.08370573073625565, MAE: 0.23859679698944092
TMIN - Loss: 0.1024654284119606, MAE: 0.3000728487968445
PRCP - Loss: 0.00017876071797218174, MAE: 0.010448559187352657
SNOW - Loss: 0.008554465137422085, MAE: 0.057786256074905396
AWND - Loss: 0.1379808634519577, MAE: 0.2428041249513626

Evaluation results: {'TMAX': {'loss': 0.08370573073625565, 'mae': 0.23859679698944092}, 'TMIN': {'loss': 0.1024654284119606, 'mae': 0.3000728487968445}, 'PRCP': {'loss': 0.00017876071797218174, 'mae': 0.010448559187352657}, 'SNOW': {'loss': 0.008554465137422085, 'mae': 0.057786256074905396}, 'AWND': {'loss': 0.1379808634519577, 'mae': 0.2428041249513626}} 
I mostly used MAE as the primary evaluation, model was performing excellent to the example data it was recieving, however even with number of added features the predictions are still off for anywhere not in CO (main goal of app was to focus on CO).

## FILE STRUCTURE
not-so-accurate-weather-app/
├── templates/
│   └── index.html
├── [weather_app.py]
├── [model.py]
├── [requirements.txt]
├── [README.md]
└── models/
    ├── [tmax_model.pkl]
    ├── [tmin_model.pkl]
    ├── [prcp_model.pkl]
    ├── [snow_model.pkl]
    └── [awnd_model.pkl]

## PROJECT CITATIONS
Deployment - Heroku
Heroku. (2024). Deployments Made Easy A Step-by-Step Guide. Retrieved from http://www.heroku.com/ 


ZIPPOPOTAM
Zippopotamus. (2024). Zip Code Galore. Retrieved from http://www.zippopotam.us/


NOAA
National Oceanic and Atmospheric Administration (NOAA). (2023). National Weather Service API. National Centers for Environmental Information. https://www.weather.gov/documentation/services-web-api


Acknowledgment Microsoft Copilot
﻿## Project: Weather Forecasting Model using Neural Networks (Project 1)
## Author: Austyn Griego
## Course: CS3210 - Machine Learning
## Date: November 23rd, 2024
## Project Title: Not So Accurate Weather App

## OPTION A -- Web Application using trained model
Option A: Build a Web App for a Machine Learning Model
Train a machine learning model on a dataset of your choice and create a web-based application to allow users to interact with it. You may use Flask or a similar web framework (here is a brief tutorial on using FlaskLinks to an external site.). This short lessonLinks to an external site. in a Microsoft online course on ML also covers how to "pickle" your model so that it can be used by your app. 

Proposal Requirements
Model Overview:
Briefly describe the machine learning model you plan to use (this can be a very basic type of model, given that you will be creating an app for it) -- NEURAL NETWORK
Include a justification for choosing this model, and point out why another type of model might not be a good choice
Dataset -- WEATHER PATTERNS, WEATHER COMPLEXITY
Describe the dataset you will use to train your model, including the source and the features. -- NOAA DATASET
App Functionality
Describe how users will interact with the model through the web app -- USER ENTERS ZIPCODE THEY WANT A FORECAST FOR
Outline any additional features (e.g., user input, visualizations, feedback) -- RETURN A FORECAST OF HIGH LOW RAIN/SNOW/AVG WIND
Tools and Technologies
Specify which frameworks or tools (e.g., Flask, HTML, CSS, etc.) you will use to build the app. -- FLASK, HTML, COPILOT, TENSFLOW

## Overview
The Not So Accurate Weather App is a web application that provides weather forecasts based on historical weather data from NOAA. The app uses Neural Network machine learning models to predict various weather parameters such as maximum temperature (TMAX), minimum temperature (TMIN), precipitation (PRCP), snowfall (SNOW), and average wind speed (AWND). The app is built using Flask and provides a user-friendly interface for entering a zipcode and viewing the (not-so-accurate) forecast.

## Features
- Predicts weather parameters: TMAX, TMIN, PRCP, SNOW, and AWND using pretrained Neural Network machine learning models.
- Displays icons and a card featuring the forecasted weather conditions.
- Styling using html and CSS, i have zero experience in these languages, however, i was able to make site look better to the user, with added colors, and background fill-in.
- Based on user inputed zipcode - app gets user longitude and lattitude to find nearby stations to gather and aggregate historical data from past 30 days and make new predictions.
- Error Handling, and debugging statements throughout code, including user informative error messages on the web application.

## Future Features 
These are the features I wanted or would still like to add outside of the project scope:
- Historical data visuals (graphs, charts, weather maps, etc.)\
- 7 Day Forecast model/visual
- More specified/useful predictions on data

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/not-so-accurate-weather-app.git
   cd not-so-accurate-weather-app
Expected Model Output: (Does not include whole terminal)

Requirements.txt has all imports required to run on venv

## Model Output
Predictions:
        TMAX      TMIN      PRCP      SNOW      AWND
0   3.518409  3.300333  0.077214  0.291762  2.072151
1   3.123149  2.364408  0.010687  0.022038  1.670326
2   3.597796  2.872427 -0.000140 -0.038106  2.421059
3   3.605472  3.144213  0.006228 -0.048102  2.119875
4   3.549781  2.926470  0.021589 -0.031075  2.506429
5   4.061012  3.577000  0.009678  0.274610  1.624122
6   3.556608  1.672168  0.000084  0.177821  1.422769
7   3.706184  2.971543 -0.001977 -0.036174  2.642668
8   3.650371  3.086612  0.010154 -0.041837  2.249779
9   3.611189  2.789905 -0.002981  0.057886  2.086437
10  3.756449  3.111994 -0.003408 -0.058065  2.335937
11  3.292114  2.266659 -0.010398  0.014362  2.029967

TMAX - Cross-validated MAE scores: [0.4149515628814697, 0.4953131377696991, 0.8685706853866577, 0.3500292897224426, 0.6028550267219543]
TMAX - Mean MAE: 0.5463439404964447, Std MAE: 0.18189619007538338

TMIN - Cross-validated MAE scores: [0.1782221645116806, 0.32035788893699646, 0.7978622317314148, 0.3646543622016907, 0.5387335419654846]
TMIN - Mean MAE: 0.43996603786945343, Std MAE: 0.21275926229873193

PRCP - Cross-validated MAE scores: [0.019182531163096428, 0.009041990153491497, 0.051679596304893494, 0.010339982807636261, 0.021017808467149734]
PRCP - Mean MAE: 0.022252381779253483, Std MAE: 0.015448810989661999

SNOW - Cross-validated MAE scores: [0.04460147023200989, 0.0419270284473896, 0.10812094062566757, 0.0446554571390152, 0.0767999142408371]
SNOW - Mean MAE: 0.06322096213698387, Std MAE: 0.025865773041333324

AWND - Cross-validated MAE scores: [0.2537437975406647, 0.24622957408428192, 0.5693408250808716, 0.21897409856319427, 0.40045130252838135]
AWND - Mean MAE: 0.33774791955947875, Std MAE: 0.13198977896209416

TMAX - Loss: 0.08370573073625565, MAE: 0.23859679698944092
TMIN - Loss: 0.1024654284119606, MAE: 0.3000728487968445
PRCP - Loss: 0.00017876071797218174, MAE: 0.010448559187352657
SNOW - Loss: 0.008554465137422085, MAE: 0.057786256074905396
AWND - Loss: 0.1379808634519577, MAE: 0.2428041249513626

Evaluation results: {'TMAX': {'loss': 0.08370573073625565, 'mae': 0.23859679698944092}, 'TMIN': {'loss': 0.1024654284119606, 'mae': 0.3000728487968445}, 'PRCP': {'loss': 0.00017876071797218174, 'mae': 0.010448559187352657}, 'SNOW': {'loss': 0.008554465137422085, 'mae': 0.057786256074905396}, 'AWND': {'loss': 0.1379808634519577, 'mae': 0.2428041249513626}} 
I mostly used MAE as the primary evaluation, model was performing excellent to the example data it was recieving, however even with number of added features the predictions are still off for anywhere not in CO (main goal of app was to focus on CO).

## FILE STRUCTURE
not-so-accurate-weather-app/
├── templates/
│   └── index.html
├── [weather_app.py]
├── [model.py]
├── [requirements.txt]
├── [README.md]
└── models/
    ├── [tmax_model.pkl]
    ├── [tmin_model.pkl]
    ├── [prcp_model.pkl]
    ├── [snow_model.pkl]
    └── [awnd_model.pkl]

## PROJECT CITATIONS
Github -- project management

Deployment - Heroku
Heroku. (2024). Deployments Made Easy A Step-by-Step Guide. Retrieved from http://www.heroku.com/ 


ZIPPOPOTAM
Zippopotamus. (2024). Zip Code Galore. Retrieved from http://www.zippopotam.us/


NOAA
National Oceanic and Atmospheric Administration (NOAA). (2023). National Weather Service API. National Centers for Environmental Information. https://www.weather.gov/documentation/services-web-api


Acknowledgment Microsoft Copilot
The development team acknowledges the use of Microsoft Copilot as an AI assistant to improve and expedite the project development process. The insights and suggestions provided by Copilot have been invaluable in achieving the project’s goals.