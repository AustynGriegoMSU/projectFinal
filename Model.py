"""
Project: Weather Forecasting Model using Neural Networks (Project 1)
Author: Austyn Griego
Course: CS3210 - Machine Learning
Date: November 23rd, 2024
Description: The following code serves to create and train a Machine learning Neural Network model to predict weather data (more specifically
TMAX: High Temp, TMIN: Low Temp, PRCP: percipitation(in), SNOW: snowfall, AWND: Average windspeed. The model is trained on historical weather 
data fetched from the NOAA API. The model is trained to predict weather based on variables: TMAX, TMIN, PRCP, SNOW, and AWND. The model is 
evaluated using the mean absolute error (MAE) metric. The code also includes feature engineering steps such as feature interactions, handling 
missing values, log transformation, and outlier removal. The code also includes cross-validation to evaluate the model's performance on unseen 
data. The code was written in Python using the TensorFlow and Keras libraries. Next steps for my project include pickling the model and creating
a web application to make weather predictions (ultimatelty in the form of a forecast) based on users inputed area code.
"""
# Import necessary libraries
import os
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow import keras
from keras import layers
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#Get Data, Concatenate Data, and Create Dataframe
# Function to fetch historical weather data from NOAA
def fetch_historical_weather_data(station_id, start_date, end_date, api_key):
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    params = {
        'datasetid': 'GHCND',
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'units': 'standard',
        'includeAttributes': 'false',
        'includeStationName': 'false',
        'includeStationLocation': 'false'
    }
    headers = {
        'token': api_key
    }
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

# Example parameters for historical weather data
station_ids = ["GHCND:USW00003017", "GHCND:USW00014742", "GHCND:USW00023183"]#station ids
start_date = "2023-01-01"  # Start date
end_date = "2023-12-31"  # End date
api_key = os.getenv('NOAA_API_TOKEN') # NOAA API --enviroment variable (sens info)

# Aggregate data from multiple stations
all_data = []
for station_id in station_ids:
    historical_data = fetch_historical_weather_data(station_id, start_date, end_date, api_key)
    if 'results' in historical_data:
        df = pd.DataFrame(historical_data['results'])
        all_data.append(df)

if all_data:
    df = pd.concat(all_data, ignore_index=True)

    # Pivot the DataFrame to make datatypes columns
    df_pivot = df.pivot_table(index='date', columns='datatype', values='value', aggfunc='first').reset_index()

    # Create aggregated weather type feature using only the available columns
    available_weather_types = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT07', 'WT08', 'WT09']
    available_weather_types = [wt for wt in available_weather_types if wt in df_pivot.columns]
    df_pivot['Adverse_Weather'] = df_pivot[available_weather_types].sum(axis=1)

    # Adjust feature selection based on available columns
    available_features = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND', 'Adverse_Weather']
    selected_features = [col for col in available_features if col in df_pivot.columns]

# BEGIN FEATURE ENGINEERING

    # Create interaction features and add them to the selected features
    if 'SNOW' in selected_features and 'TMIN' in selected_features:
        df_pivot['SNOW_TMIN'] = df_pivot['SNOW'] * df_pivot['TMIN']
        selected_features.append('SNOW_TMIN')
    if 'PRCP' in selected_features and 'TMAX' in selected_features:
        df_pivot['PRCP_TMAX'] = df_pivot['PRCP'] * df_pivot['TMAX']
        selected_features.append('PRCP_TMAX')
    if 'PRCP' in selected_features and 'AWND' in selected_features:
        df_pivot['PRCP_AWND'] = df_pivot['PRCP'] * df_pivot['AWND']
        selected_features.append('PRCP_AWND')
    if 'SNOW' in selected_features and 'AWND' in selected_features:
        df_pivot['SNOW_AWND'] = df_pivot['SNOW'] * df_pivot['AWND']
        selected_features.append('SNOW_AWND')
    if 'TMAX' in selected_features and 'AWND' in selected_features:
        df_pivot['TMAX_AWND'] = df_pivot['TMAX'] * df_pivot['AWND']
        selected_features.append('TMAX_AWND')
    if 'TMIN' in selected_features and 'AWND' in selected_features:
        df_pivot['TMIN_AWND'] = df_pivot['TMIN'] * df_pivot['AWND']
        selected_features.append('TMIN_AWND')  

    print("Selected features including interactions:", selected_features)

    # Handle missing values in selected features
    # Check for missing values
    print("Missing values in selected features:")
    print(df_pivot[selected_features].isnull().sum())
    
    # Handle missing values in selected features: Fill missing values with the mean
    df_pivot[selected_features] = df_pivot[selected_features].fillna(df_pivot[selected_features].mean())
    
    # Apply log transformation to skewed features
    df_pivot['TMAX'] = np.log1p(df_pivot['TMAX'])
    df_pivot['TMIN'] = np.log1p(df_pivot['TMIN'])
    df_pivot['PRCP'] = np.log1p(df_pivot['PRCP'])
    df_pivot['SNOW'] = np.log1p(df_pivot['SNOW'])
    df_pivot['AWND'] = np.log1p(df_pivot['AWND'])

    # Remove outliers in TMAX and TMIN using IQR method
    # Calculate IQR for TMIN
    Q1_tmin = df_pivot['TMIN'].quantile(0.25)
    Q3_tmin = df_pivot['TMIN'].quantile(0.75)
    IQR_tmin = Q3_tmin - Q1_tmin

    # Calculate lower and upper bounds for TMIN
    lower_bound_tmin = Q1_tmin - 1.5 * IQR_tmin
    upper_bound_tmin = Q3_tmin + 1.5 * IQR_tmin

    # Calculate IQR for TMAX
    Q1_tmax = df_pivot['TMAX'].quantile(0.25)
    Q3_tmax = df_pivot['TMAX'].quantile(0.75)
    IQR_tmax = Q3_tmax - Q1_tmax

    # Calculate lower and upper bounds for TMAX
    lower_bound_tmax = Q1_tmax - 1.5 * IQR_tmax
    upper_bound_tmax = Q3_tmax + 1.5 * IQR_tmax

    # Remove outliers from TMIN
    df_pivot = df_pivot[(df_pivot['TMIN'] >= lower_bound_tmin) & (df_pivot['TMIN'] <= upper_bound_tmin)]
    # Remove outliers from TMAX
    df_pivot = df_pivot[(df_pivot['TMAX'] >= lower_bound_tmax) & (df_pivot['TMAX'] <= upper_bound_tmax)]
    print("Outliers removed from TMIN and TMAX")

# END FEATURE ENGINEERING
# NEURAL NETWORK MODEL CREATION/TRAINING/EVALUATION

    # Define features and targets
    X = df_pivot[selected_features]
    y_temp_max = df_pivot['TMAX']
    y_temp_min = df_pivot['TMIN']
    y_precip = df_pivot['PRCP']
    y_snow = df_pivot['SNOW']
    y_wind = df_pivot['AWND']

    # Split data into training and testing sets
    X_train, X_test, y_train_temp_max, y_test_temp_max = train_test_split(X, y_temp_max, test_size=0.2, random_state=42)
    X_train, X_test, y_train_temp_min, y_test_temp_min = train_test_split(X, y_temp_min, test_size=0.2, random_state=42)
    X_train, X_test, y_train_precip, y_test_precip = train_test_split(X, y_precip, test_size=0.2, random_state=42)
    X_train, X_test, y_train_snow, y_test_snow = train_test_split(X, y_snow, test_size=0.2, random_state=42)
    X_train, X_test, y_train_wind, y_test_wind = train_test_split(X, y_wind, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for app later
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Defines a function to create a neural network model as needed
    #I went this route in order to easily create Individual models for each target variable, ie SNOW DATA to PREDICT SNOWFALL)
    def create_nn_model(input_shape):
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Train models for each target variable
    nn_models = {}

    # Creates a NN for TMAX
    nn_models['TMAX'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['TMAX'].fit(X_train_scaled, y_train_temp_max, epochs=100, validation_split=0.2)

    # Creates a NN for TMIN
    nn_models['TMIN'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['TMIN'].fit(X_train_scaled, y_train_temp_min, epochs=100, validation_split=0.2)

    # Creates a NN for PRCP
    nn_models['PRCP'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['PRCP'].fit(X_train_scaled, y_train_precip, epochs=100, validation_split=0.2)

    # Creates a NN for SNOW
    nn_models['SNOW'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['SNOW'].fit(X_train_scaled, y_train_snow, epochs=100, validation_split=0.2)

    # Creates a NN for AWND
    nn_models['AWND'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['AWND'].fit(X_train_scaled, y_train_wind, epochs=100, validation_split=0.2)

    # Evaluate models
    # Initialize evaluation results
    evaluation_results = {}
    #Evaluation Results for each model
    evaluation_results['TMAX'] = nn_models['TMAX'].evaluate(X_test_scaled, y_test_temp_max)
    evaluation_results['TMIN'] = nn_models['TMIN'].evaluate(X_test_scaled, y_test_temp_min)
    evaluation_results['PRCP'] = nn_models['PRCP'].evaluate(X_test_scaled, y_test_precip)
    evaluation_results['SNOW'] = nn_models['SNOW'].evaluate(X_test_scaled, y_test_snow)
    evaluation_results['AWND'] = nn_models['AWND'].evaluate(X_test_scaled, y_test_wind)

    print(f"Evaluation results: {evaluation_results}")

    # Make predictions for each target variable
    # Initialize predictions results
    predictions = {}

    predictions['TMAX'] = nn_models['TMAX'].predict(X_test_scaled).flatten()
    predictions['TMIN'] = nn_models['TMIN'].predict(X_test_scaled).flatten()
    predictions['PRCP'] = nn_models['PRCP'].predict(X_test_scaled).flatten()
    predictions['SNOW'] = nn_models['SNOW'].predict(X_test_scaled).flatten()
    predictions['AWND'] = nn_models['AWND'].predict(X_test_scaled).flatten()

    # Combine predictions into a DataFrame and print the results
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)

    # Evaluate models and print loss and MAE values
    evaluation_results = {}
    
    # Begin Cross Validation Section
    # Defines a function to cross-validate each model
    def cross_validate_model(X, y, model_name):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        X_scaled = scaler.fit_transform(X)
        cvscores = []

        for train, test in kfold.split(X_scaled, y):
            model = create_nn_model(X_scaled.shape[1])
            model.fit(X_scaled[train], y.iloc[train], epochs=100, validation_data=(X_scaled[test], y.iloc[test]), verbose=0)

            #Evaluate the model
            scores = model.evaluate(X_scaled[test], y.iloc[test], verbose=0)
            cvscores.append(scores[1])  # MAE is the second metric
        #Print the results for each model for cross-validated MAE scores and mean and Standard Dev. of MAE
        print(f"{model_name} - Cross-validated MAE scores: {cvscores}")
        print(f"{model_name} - Mean MAE: {np.mean(cvscores)}, Std MAE: {np.std(cvscores)}")

    # Define features and targets
    X = df_pivot[selected_features]
    targets = {
        'TMAX': df_pivot['TMAX'],
        'TMIN': df_pivot['TMIN'],
        'PRCP': df_pivot['PRCP'],
        'SNOW': df_pivot['SNOW'],
        'AWND': df_pivot['AWND']
    }
    #apply the cross_validate_model function to each target variable
    for model_name, y in targets.items():
        cross_validate_model(X, y, model_name)
    # End Cross Validation section
    
    # Evaluate models on the test set
    # Prints loss and MAE for TMAX
    loss, mae = nn_models['TMAX'].evaluate(X_test_scaled, y_test_temp_max, verbose=0)
    evaluation_results['TMAX'] = {'loss': loss, 'mae': mae}
    print(f"TMAX - Loss: {loss}, MAE: {mae}")

    # Pickle the model TMAX
    with open('TMAX_model.pkl', 'wb') as f:
        pickle.dump(nn_models['TMAX'], f)
    
    # Prints loss and MAE for TMIN
    loss, mae = nn_models['TMIN'].evaluate(X_test_scaled, y_test_temp_min, verbose=0)
    evaluation_results['TMIN'] = {'loss': loss, 'mae': mae}
    print(f"TMIN - Loss: {loss}, MAE: {mae}")

    # Pickle the model TMIN
    with open('TMIN_model.pkl', 'wb') as f:
        pickle.dump(nn_models['TMIN'], f)

    # Prints loss and MAE for PRCP
    loss, mae = nn_models['PRCP'].evaluate(X_test_scaled, y_test_precip, verbose=0)
    evaluation_results['PRCP'] = {'loss': loss, 'mae': mae}
    print(f"PRCP - Loss: {loss}, MAE: {mae}")
    
    # Pickle the model PRCP
    with open('PRCP_model.pkl', 'wb') as f:
        pickle.dump(nn_models['PRCP'], f)

    # Prints loss and MAE for SNOW
    loss, mae = nn_models['SNOW'].evaluate(X_test_scaled, y_test_snow, verbose=0)
    evaluation_results['SNOW'] = {'loss': loss, 'mae': mae}
    print(f"SNOW - Loss: {loss}, MAE: {mae}")

    # Pickle the model SNOW
    with open('SNOW_model.pkl', 'wb') as f:
        pickle.dump(nn_models['SNOW'], f)

    # Prints loss and MAE for AWND
    loss, mae = nn_models['AWND'].evaluate(X_test_scaled, y_test_wind, verbose=0)
    evaluation_results['AWND'] = {'loss': loss, 'mae': mae}
    print(f"AWND - Loss: {loss}, MAE: {mae}")

    # Pickle the model AWND
    with open('AWND_model.pkl', 'wb') as f:
        pickle.dump(nn_models['AWND'], f)

    # Print the Overall summary of evaluation results
    print(f"Evaluation results: {evaluation_results}")

else:
    #This would be the case if there are no results found in the API responses fetched
    print("No results found in the API responses.")

# END OF CODE
"""
Project: Weather Forecasting Model using Neural Networks (Project 1)
Author: Austyn Griego
Course: CS3210 - Machine Learning
Date: November 23rd, 2024
Description: The following code serves to create and train a Machine learning Neural Network model to predict weather data (more specifically
TMAX: High Temp, TMIN: Low Temp, PRCP: percipitation(in), SNOW: snowfall, AWND: Average windspeed. The model is trained on historical weather 
data fetched from the NOAA API. The model is trained to predict weather based on variables: TMAX, TMIN, PRCP, SNOW, and AWND. The model is 
evaluated using the mean absolute error (MAE) metric. The code also includes feature engineering steps such as feature interactions, handling 
missing values, log transformation, and outlier removal. The code also includes cross-validation to evaluate the model's performance on unseen 
data. The code was written in Python using the TensorFlow and Keras libraries. Next steps for my project include pickling the model and creating
a web application to make weather predictions (ultimatelty in the form of a forecast) based on users inputed area code.
"""
# Import necessary libraries
import os
import requests
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from tensorflow import keras
from keras import layers
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

#Get Data, Concatenate Data, and Create Dataframe
# Function to fetch historical weather data from NOAA
def fetch_historical_weather_data(station_id, start_date, end_date, api_key):
    base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"
    params = {
        'datasetid': 'GHCND',
        'stationid': station_id,
        'startdate': start_date,
        'enddate': end_date,
        'limit': 1000,
        'units': 'standard',
        'includeAttributes': 'false',
        'includeStationName': 'false',
        'includeStationLocation': 'false'
    }
    headers = {
        'token': api_key
    }
    response = requests.get(base_url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error: {response.status_code}")

# Example parameters for historical weather data
station_ids = ["GHCND:USW00003017", "GHCND:USW00014742", "GHCND:USW00023183"]#station ids
start_date = "2023-01-01"  # Start date
end_date = "2023-12-31"  # End date
api_key = os.getenv('NOAA_API_TOKEN') # NOAA API --enviroment variable (sens info)

# Aggregate data from multiple stations
all_data = []
for station_id in station_ids:
    historical_data = fetch_historical_weather_data(station_id, start_date, end_date, api_key)
    if 'results' in historical_data:
        df = pd.DataFrame(historical_data['results'])
        all_data.append(df)

if all_data:
    df = pd.concat(all_data, ignore_index=True)

    # Pivot the DataFrame to make datatypes columns
    df_pivot = df.pivot_table(index='date', columns='datatype', values='value', aggfunc='first').reset_index()

    # Create aggregated weather type feature using only the available columns
    available_weather_types = ['WT01', 'WT02', 'WT03', 'WT04', 'WT05', 'WT06', 'WT07', 'WT08', 'WT09']
    available_weather_types = [wt for wt in available_weather_types if wt in df_pivot.columns]
    df_pivot['Adverse_Weather'] = df_pivot[available_weather_types].sum(axis=1)

    # Adjust feature selection based on available columns
    available_features = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND', 'Adverse_Weather']
    selected_features = [col for col in available_features if col in df_pivot.columns]

# BEGIN FEATURE ENGINEERING

    # Create interaction features and add them to the selected features
    if 'SNOW' in selected_features and 'TMIN' in selected_features:
        df_pivot['SNOW_TMIN'] = df_pivot['SNOW'] * df_pivot['TMIN']
        selected_features.append('SNOW_TMIN')
    if 'PRCP' in selected_features and 'TMAX' in selected_features:
        df_pivot['PRCP_TMAX'] = df_pivot['PRCP'] * df_pivot['TMAX']
        selected_features.append('PRCP_TMAX')
    if 'PRCP' in selected_features and 'AWND' in selected_features:
        df_pivot['PRCP_AWND'] = df_pivot['PRCP'] * df_pivot['AWND']
        selected_features.append('PRCP_AWND')
    if 'SNOW' in selected_features and 'AWND' in selected_features:
        df_pivot['SNOW_AWND'] = df_pivot['SNOW'] * df_pivot['AWND']
        selected_features.append('SNOW_AWND')
    if 'TMAX' in selected_features and 'AWND' in selected_features:
        df_pivot['TMAX_AWND'] = df_pivot['TMAX'] * df_pivot['AWND']
        selected_features.append('TMAX_AWND')
    if 'TMIN' in selected_features and 'AWND' in selected_features:
        df_pivot['TMIN_AWND'] = df_pivot['TMIN'] * df_pivot['AWND']
        selected_features.append('TMIN_AWND')  

    print("Selected features including interactions:", selected_features)

    # Handle missing values in selected features
    # Check for missing values
    print("Missing values in selected features:")
    print(df_pivot[selected_features].isnull().sum())
    
    # Handle missing values in selected features: Fill missing values with the mean
    df_pivot[selected_features] = df_pivot[selected_features].fillna(df_pivot[selected_features].mean())
    
    # Apply log transformation to skewed features
    df_pivot['TMAX'] = np.log1p(df_pivot['TMAX'])
    df_pivot['TMIN'] = np.log1p(df_pivot['TMIN'])
    df_pivot['PRCP'] = np.log1p(df_pivot['PRCP'])
    df_pivot['SNOW'] = np.log1p(df_pivot['SNOW'])
    df_pivot['AWND'] = np.log1p(df_pivot['AWND'])

    # Remove outliers in TMAX and TMIN using IQR method
    # Calculate IQR for TMIN
    Q1_tmin = df_pivot['TMIN'].quantile(0.25)
    Q3_tmin = df_pivot['TMIN'].quantile(0.75)
    IQR_tmin = Q3_tmin - Q1_tmin

    # Calculate lower and upper bounds for TMIN
    lower_bound_tmin = Q1_tmin - 1.5 * IQR_tmin
    upper_bound_tmin = Q3_tmin + 1.5 * IQR_tmin

    # Calculate IQR for TMAX
    Q1_tmax = df_pivot['TMAX'].quantile(0.25)
    Q3_tmax = df_pivot['TMAX'].quantile(0.75)
    IQR_tmax = Q3_tmax - Q1_tmax

    # Calculate lower and upper bounds for TMAX
    lower_bound_tmax = Q1_tmax - 1.5 * IQR_tmax
    upper_bound_tmax = Q3_tmax + 1.5 * IQR_tmax

    # Remove outliers from TMIN
    df_pivot = df_pivot[(df_pivot['TMIN'] >= lower_bound_tmin) & (df_pivot['TMIN'] <= upper_bound_tmin)]
    # Remove outliers from TMAX
    df_pivot = df_pivot[(df_pivot['TMAX'] >= lower_bound_tmax) & (df_pivot['TMAX'] <= upper_bound_tmax)]
    print("Outliers removed from TMIN and TMAX")

# END FEATURE ENGINEERING
# NEURAL NETWORK MODEL CREATION/TRAINING/EVALUATION

    # Define features and targets
    X = df_pivot[selected_features]
    y_temp_max = df_pivot['TMAX']
    y_temp_min = df_pivot['TMIN']
    y_precip = df_pivot['PRCP']
    y_snow = df_pivot['SNOW']
    y_wind = df_pivot['AWND']

    # Split data into training and testing sets
    X_train, X_test, y_train_temp_max, y_test_temp_max = train_test_split(X, y_temp_max, test_size=0.2, random_state=42)
    X_train, X_test, y_train_temp_min, y_test_temp_min = train_test_split(X, y_temp_min, test_size=0.2, random_state=42)
    X_train, X_test, y_train_precip, y_test_precip = train_test_split(X, y_precip, test_size=0.2, random_state=42)
    X_train, X_test, y_train_snow, y_test_snow = train_test_split(X, y_snow, test_size=0.2, random_state=42)
    X_train, X_test, y_train_wind, y_test_wind = train_test_split(X, y_wind, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler for app later
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Defines a function to create a neural network model as needed
    #I went this route in order to easily create Individual models for each target variable, ie SNOW DATA to PREDICT SNOWFALL)
    def create_nn_model(input_shape):
        model = keras.Sequential([
            layers.Input(shape=(input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model

    # Train models for each target variable
    nn_models = {}

    # Creates a NN for TMAX
    nn_models['TMAX'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['TMAX'].fit(X_train_scaled, y_train_temp_max, epochs=100, validation_split=0.2)

    # Creates a NN for TMIN
    nn_models['TMIN'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['TMIN'].fit(X_train_scaled, y_train_temp_min, epochs=100, validation_split=0.2)

    # Creates a NN for PRCP
    nn_models['PRCP'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['PRCP'].fit(X_train_scaled, y_train_precip, epochs=100, validation_split=0.2)

    # Creates a NN for SNOW
    nn_models['SNOW'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['SNOW'].fit(X_train_scaled, y_train_snow, epochs=100, validation_split=0.2)

    # Creates a NN for AWND
    nn_models['AWND'] = create_nn_model(X_train_scaled.shape[1])
    nn_models['AWND'].fit(X_train_scaled, y_train_wind, epochs=100, validation_split=0.2)

    # Evaluate models
    # Initialize evaluation results
    evaluation_results = {}
    #Evaluation Results for each model
    evaluation_results['TMAX'] = nn_models['TMAX'].evaluate(X_test_scaled, y_test_temp_max)
    evaluation_results['TMIN'] = nn_models['TMIN'].evaluate(X_test_scaled, y_test_temp_min)
    evaluation_results['PRCP'] = nn_models['PRCP'].evaluate(X_test_scaled, y_test_precip)
    evaluation_results['SNOW'] = nn_models['SNOW'].evaluate(X_test_scaled, y_test_snow)
    evaluation_results['AWND'] = nn_models['AWND'].evaluate(X_test_scaled, y_test_wind)

    print(f"Evaluation results: {evaluation_results}")

    # Make predictions for each target variable
    # Initialize predictions results
    predictions = {}

    predictions['TMAX'] = nn_models['TMAX'].predict(X_test_scaled).flatten()
    predictions['TMIN'] = nn_models['TMIN'].predict(X_test_scaled).flatten()
    predictions['PRCP'] = nn_models['PRCP'].predict(X_test_scaled).flatten()
    predictions['SNOW'] = nn_models['SNOW'].predict(X_test_scaled).flatten()
    predictions['AWND'] = nn_models['AWND'].predict(X_test_scaled).flatten()

    # Combine predictions into a DataFrame and print the results
    predictions_df = pd.DataFrame(predictions)
    print(predictions_df)

    # Evaluate models and print loss and MAE values
    evaluation_results = {}
    
    # Begin Cross Validation Section
    # Defines a function to cross-validate each model
    def cross_validate_model(X, y, model_name):
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        X_scaled = scaler.fit_transform(X)
        cvscores = []

        for train, test in kfold.split(X_scaled, y):
            model = create_nn_model(X_scaled.shape[1])
            model.fit(X_scaled[train], y.iloc[train], epochs=100, validation_data=(X_scaled[test], y.iloc[test]), verbose=0)

            #Evaluate the model
            scores = model.evaluate(X_scaled[test], y.iloc[test], verbose=0)
            cvscores.append(scores[1])  # MAE is the second metric
        #Print the results for each model for cross-validated MAE scores and mean and Standard Dev. of MAE
        print(f"{model_name} - Cross-validated MAE scores: {cvscores}")
        print(f"{model_name} - Mean MAE: {np.mean(cvscores)}, Std MAE: {np.std(cvscores)}")

    # Define features and targets
    X = df_pivot[selected_features]
    targets = {
        'TMAX': df_pivot['TMAX'],
        'TMIN': df_pivot['TMIN'],
        'PRCP': df_pivot['PRCP'],
        'SNOW': df_pivot['SNOW'],
        'AWND': df_pivot['AWND']
    }
    #apply the cross_validate_model function to each target variable
    for model_name, y in targets.items():
        cross_validate_model(X, y, model_name)
    # End Cross Validation section
    
    # Evaluate models on the test set
    # Prints loss and MAE for TMAX
    loss, mae = nn_models['TMAX'].evaluate(X_test_scaled, y_test_temp_max, verbose=0)
    evaluation_results['TMAX'] = {'loss': loss, 'mae': mae}
    print(f"TMAX - Loss: {loss}, MAE: {mae}")

    # Pickle the model TMAX
    with open('TMAX_model.pkl', 'wb') as f:
        pickle.dump(nn_models['TMAX'], f)
    
    # Prints loss and MAE for TMIN
    loss, mae = nn_models['TMIN'].evaluate(X_test_scaled, y_test_temp_min, verbose=0)
    evaluation_results['TMIN'] = {'loss': loss, 'mae': mae}
    print(f"TMIN - Loss: {loss}, MAE: {mae}")

    # Pickle the model TMIN
    with open('TMIN_model.pkl', 'wb') as f:
        pickle.dump(nn_models['TMIN'], f)

    # Prints loss and MAE for PRCP
    loss, mae = nn_models['PRCP'].evaluate(X_test_scaled, y_test_precip, verbose=0)
    evaluation_results['PRCP'] = {'loss': loss, 'mae': mae}
    print(f"PRCP - Loss: {loss}, MAE: {mae}")
    
    # Pickle the model PRCP
    with open('PRCP_model.pkl', 'wb') as f:
        pickle.dump(nn_models['PRCP'], f)

    # Prints loss and MAE for SNOW
    loss, mae = nn_models['SNOW'].evaluate(X_test_scaled, y_test_snow, verbose=0)
    evaluation_results['SNOW'] = {'loss': loss, 'mae': mae}
    print(f"SNOW - Loss: {loss}, MAE: {mae}")

    # Pickle the model SNOW
    with open('SNOW_model.pkl', 'wb') as f:
        pickle.dump(nn_models['SNOW'], f)

    # Prints loss and MAE for AWND
    loss, mae = nn_models['AWND'].evaluate(X_test_scaled, y_test_wind, verbose=0)
    evaluation_results['AWND'] = {'loss': loss, 'mae': mae}
    print(f"AWND - Loss: {loss}, MAE: {mae}")

    # Pickle the model AWND
    with open('AWND_model.pkl', 'wb') as f:
        pickle.dump(nn_models['AWND'], f)

    # Print the Overall summary of evaluation results
    print(f"Evaluation results: {evaluation_results}")

else:
    #This would be the case if there are no results found in the API responses fetched
    print("No results found in the API responses.")

# END OF CODE
#Lower MAE is better ie low MAE for TMAX such as 5 means the model is off by 5 degrees on average