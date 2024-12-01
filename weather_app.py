import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, render_template
import pickle
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'secret'
app.config['SESSION_TYPE'] = 'filesystem'


# Load pickled models and scaler
with open('tmax_model.pkl', 'rb') as f:
    tmax_model = pickle.load(f)
with open('tmin_model.pkl', 'rb') as f:
    tmin_model = pickle.load(f)
with open('prcp_model.pkl', 'rb') as f:
    prcp_model = pickle.load(f)
with open('snow_model.pkl', 'rb') as f:
    snow_model = pickle.load(f)
with open('awnd_model.pkl', 'rb') as f:
    awnd_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# NOAA API TOKEN to be used throughout app -- set as env variable to keep sens data safe
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN')

# Function to get Lat and Lon from Users zipcode
def get_lat_lon(zipcode):
    #API zippopotam to get Lat/lon from zip
    response = requests.get(f'http://api.zippopotam.us/us/{zipcode}')
    #Error handling
    if response.status_code != 200:
        #print(f"Error fetching geolocation data for zipcode {zipcode}: {response.status_code}")
        return None, None, None, None
    data = response.json()
    if 'places' not in data or len(data['places']) == 0:
        #print(f"No geolocation data found for zipcode {zipcode}")
        return None, None, None, None
    #Found zip Lat and lon - return Zip, City, State in app*
    place = data['places'][0]
    return float(place['latitude']), float(place['longitude']), place['place name'], place['state abbreviation']

# Function to fetch nearby station data based on lat and lon
def fetch_nearby_stations(lat, lon):
    headers = {
        'token': NOAA_API_TOKEN
    }
    params = {
        'datasetid': 'GHCND',
        'extent': f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}",
        'limit': 100,
        'sortfield': 'datacoverage',
        'sortorder': 'desc'
    }
    # Error Handling
    response = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/stations', headers=headers, params=params)
    if response.status_code != 200:
        #print(f"Error fetching stations from NOAA API: {response.status_code}")
        #print(response.text)
        return []
    data = response.json()
    if 'results' not in data:
        #print("No 'results' key in the response data")
        #print(data)
        return []
    # Found data - return NOAA stations
    return data['results']

# Function to fetch historical data from selected stations
def fetch_historical_data(lat, lon):
    stations = fetch_nearby_stations(lat, lon)
    station_ids = [station['id'] for station in stations if 'GHCND' in station['id']]
    # Last 30 days of data used as input data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    headers = {
        'token': NOAA_API_TOKEN
    }
    # initialize empty list for historical data, to be added to data frame
    historical_data = []
    stations_with_data = 0
    for station_id in station_ids:
        if stations_with_data >= 15:
            break
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': start_date_str,
            'enddate': end_date_str,
            'units': 'standard',
            'limit': 1000,
            'includemetadata': 'false'
        }
        # Collects data from station data -- from NOAA API
        response = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data', headers=headers, params=params)
        # Error Handling
        if response.status_code != 200:
            #print(f"Error fetching data from NOAA API for station {station_id}: {response.status_code}")
            #print(response.text)
            continue
        data = response.json()
        if 'results' not in data:
            #print(f"No 'results' key in the response data for station {station_id}")
            #print(data)
            continue
        # Adds new data to historical_data list
        for record in data['results']:
            historical_data.append({
                'date': record['date'],
                'datatype': record['datatype'],
                'value': record['value']
            })
        # Add 1 to station count -- looking for 15 or more to get sufficient data
        stations_with_data += 1
    # Return an empty DataFrame if no data is fetched
    if not historical_data:
        return pd.DataFrame()
    
    # historical_data lst added to dataframe 
    df = pd.DataFrame(historical_data)
    # Datetime formatting
    df['date'] = pd.to_datetime(df['date'])
    # Aggregate data by date and datatype
    df = df.groupby(['date', 'datatype']).agg({'value': 'mean'}).reset_index()
    # Pivot the DataFrame
    df = df.pivot(index='date', columns='datatype', values='value').reset_index()

    # Ensure all necessary features are present (TMAX,TMIN,SNOW,PRCP,AWND)
    required_features = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Fill missing features with 0

    # Create interaction features and add them to the DataFrame
    if 'SNOW' in df.columns and 'TMIN' in df.columns:
        df['SNOW_TMIN'] = df['SNOW'] * df['TMIN']
    if 'PRCP' in df.columns and 'TMAX' in df.columns:
        df['PRCP_TMAX'] = df['PRCP'] * df['TMAX']
    if 'PRCP' in df.columns and 'AWND' in df.columns:
        df['PRCP_AWND'] = df['PRCP'] * df['AWND']
    if 'SNOW' in df.columns and 'AWND' in df.columns:
        df['SNOW_AWND'] = df['SNOW'] * df['AWND']
    if 'TMAX' in df.columns and 'AWND' in df.columns:
        df['TMAX_AWND'] = df['TMAX'] * df['AWND']
    if 'TMIN' in df.columns and 'AWND' in df.columns:
        df['TMIN_AWND'] = df['TMIN'] * df['AWND']

    # Added Additional features to match the expected input shape (12 outputs)
    additional_features = ['Adverse_Weather', 'SNOW_TMIN', 'PRCP_TMAX', 'PRCP_AWND', 'SNOW_AWND', 'TMAX_AWND', 'TMIN_AWND']
    for feature in additional_features:
        if feature not in df.columns:
            df[feature] = 0  # Replace missing additonal features with 0

    # Apply log transformation to skewed features (reduce skewness)
    df['TMAX'] = np.log1p(df['TMAX'])
    df['TMIN'] = np.log1p(df['TMIN'])
    df['PRCP'] = np.log1p(df['PRCP'])
    df['SNOW'] = np.log1p(df['SNOW'])
    df['AWND'] = np.log1p(df['AWND'])

    # Ensure the feature names and order match those used during training
    # Ordered: TMAX,TMIN,PRCP,SNOW,AWND,Adverse,SNOW_TMIN,PRCP_TMAX,PRCP_AWND,SNOW_AWND,TMAX_AWND,TMIN_AWND
    all_features = required_features + additional_features
    df = df[all_features]

    # Prints Station Ids with correct datatypes
    #print(f"Historical data for stations {station_ids[:stations_with_data]}")
    return df

# Function to make predictions based on users zipcode
def predict_forecast(zipcode):
    lat, lon, city, state = get_lat_lon(zipcode)
    # Error Handling
    if lat is None or lon is None:
        return {"error": "Unable to fetch geolocation data for the provided zipcode"}

    historical_data = fetch_historical_data(lat, lon)
    if historical_data.empty:
        return {"error": "No historical data available for the provided location"}

    # Print first few rows of data frame (BEFORE SCALING)
    #print("Historical data before scaling:")
    #print(historical_data.head())

    # Scale the data
    X_scaled = scaler.transform(historical_data)

    # Ensure the scaled data is in the correct format
    #print("Scaled data:")
    #print(X_scaled[:5])

    # Make predictions with scaling
    tmax_pred = tmax_model.predict(X_scaled)
    tmin_pred = tmin_model.predict(X_scaled)
    prcp_pred = prcp_model.predict(X_scaled)
    snow_pred = snow_model.predict(X_scaled)
    awnd_pred = awnd_model.predict(X_scaled)

    # Ensure the predictions are in the correct format
    #print("Predictions:")
    #print("TMAX:", tmax_pred[:5])
    #print("TMIN:", tmin_pred[:5])
    #print("PRCP:", prcp_pred[:5])
    #print("SNOW:", snow_pred[:5])
    #print("AWND:", awnd_pred[:5])


    # Convert TMAX and TMIN from Celsius to Fahrenheit (TMIN additional scaling)
    tmax_f = (np.mean(tmax_pred) * 9/5) + 32
    tmin_f = ((np.mean(tmin_pred) * 9/5) + 32)* 0.6

    forecast = {
        'TMAX': float(tmax_f),  # Use mean of TMAX predictions
        'TMIN': float(tmin_f),  # Use mean of TMIN predictions
        'PRCP': float(np.mean(prcp_pred)),  # Use mean of PRCP predicitons
        'SNOW': float(np.mean(snow_pred)),  # Use mean of SNOW predictions  
        'AWND': float(np.mean(awnd_pred)),  # Use mean of AWND predictions
        'city': city,
        'state': state
    }
    return forecast

# Handles GET and POST requests 
@app.route('/', methods=['GET', 'POST'])
# Function to access/execute index.html
def index():
    if request.method == 'POST':
        # Extract Zipcode from form submission
        zipcode = request.form['zipcode']
        # call predict_forecast to get forecast
        forecast = predict_forecast(zipcode)
        # Renders web app using index.html to return predictions to user
        return render_template('index.html', forecast=forecast, zipcode=zipcode)
    # for GET, simply renders empty index.html template on webapp
    return render_template('index.html')

if __name__ == "__main__":
import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, request, render_template
import pickle
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask application
app = Flask(__name__)
app.secret_key = 'secret'
app.config['SESSION_TYPE'] = 'filesystem'


# Load pickled models and scaler
with open('tmax_model.pkl', 'rb') as f:
    tmax_model = pickle.load(f)
with open('tmin_model.pkl', 'rb') as f:
    tmin_model = pickle.load(f)
with open('prcp_model.pkl', 'rb') as f:
    prcp_model = pickle.load(f)
with open('snow_model.pkl', 'rb') as f:
    snow_model = pickle.load(f)
with open('awnd_model.pkl', 'rb') as f:
    awnd_model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# NOAA API TOKEN to be used throughout app -- set as env variable to keep sens data safe
NOAA_API_TOKEN = os.getenv('NOAA_API_TOKEN')

# Function to get Lat and Lon from Users zipcode
def get_lat_lon(zipcode):
    #API zippopotam to get Lat/lon from zip
    response = requests.get(f'http://api.zippopotam.us/us/{zipcode}')
    #Error handling
    if response.status_code != 200:
        #print(f"Error fetching geolocation data for zipcode {zipcode}: {response.status_code}")
        return None, None, None, None
    data = response.json()
    if 'places' not in data or len(data['places']) == 0:
        #print(f"No geolocation data found for zipcode {zipcode}")
        return None, None, None, None
    #Found zip Lat and lon - return Zip, City, State in app*
    place = data['places'][0]
    return float(place['latitude']), float(place['longitude']), place['place name'], place['state abbreviation']

# Function to fetch nearby station data based on lat and lon
def fetch_nearby_stations(lat, lon):
    headers = {
        'token': NOAA_API_TOKEN
    }
    params = {
        'datasetid': 'GHCND',
        'extent': f"{lat-0.5},{lon-0.5},{lat+0.5},{lon+0.5}",
        'limit': 100,
        'sortfield': 'datacoverage',
        'sortorder': 'desc'
    }
    # Error Handling
    response = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/stations', headers=headers, params=params)
    if response.status_code != 200:
        #print(f"Error fetching stations from NOAA API: {response.status_code}")
        #print(response.text)
        return []
    data = response.json()
    if 'results' not in data:
        #print("No 'results' key in the response data")
        #print(data)
        return []
    # Found data - return NOAA stations
    return data['results']

# Function to fetch historical data from selected stations
def fetch_historical_data(lat, lon):
    stations = fetch_nearby_stations(lat, lon)
    station_ids = [station['id'] for station in stations if 'GHCND' in station['id']]
    # Last 30 days of data used as input data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    headers = {
        'token': NOAA_API_TOKEN
    }
    # initialize empty list for historical data, to be added to data frame
    historical_data = []
    stations_with_data = 0
    for station_id in station_ids:
        if stations_with_data >= 15:
            break
        params = {
            'datasetid': 'GHCND',
            'stationid': station_id,
            'startdate': start_date_str,
            'enddate': end_date_str,
            'units': 'standard',
            'limit': 1000,
            'includemetadata': 'false'
        }
        # Collects data from station data -- from NOAA API
        response = requests.get('https://www.ncdc.noaa.gov/cdo-web/api/v2/data', headers=headers, params=params)
        # Error Handling
        if response.status_code != 200:
            #print(f"Error fetching data from NOAA API for station {station_id}: {response.status_code}")
            #print(response.text)
            continue
        data = response.json()
        if 'results' not in data:
            #print(f"No 'results' key in the response data for station {station_id}")
            #print(data)
            continue
        # Adds new data to historical_data list
        for record in data['results']:
            historical_data.append({
                'date': record['date'],
                'datatype': record['datatype'],
                'value': record['value']
            })
        # Add 1 to station count -- looking for 15 or more to get sufficient data
        stations_with_data += 1
    # Return an empty DataFrame if no data is fetched
    if not historical_data:
        return pd.DataFrame()
    
    # historical_data lst added to dataframe 
    df = pd.DataFrame(historical_data)
    # Datetime formatting
    df['date'] = pd.to_datetime(df['date'])
    # Aggregate data by date and datatype
    df = df.groupby(['date', 'datatype']).agg({'value': 'mean'}).reset_index()
    # Pivot the DataFrame
    df = df.pivot(index='date', columns='datatype', values='value').reset_index()

    # Ensure all necessary features are present (TMAX,TMIN,SNOW,PRCP,AWND)
    required_features = ['TMAX', 'TMIN', 'PRCP', 'SNOW', 'AWND']
    for feature in required_features:
        if feature not in df.columns:
            df[feature] = 0  # Fill missing features with 0

    # Create interaction features and add them to the DataFrame
    if 'SNOW' in df.columns and 'TMIN' in df.columns:
        df['SNOW_TMIN'] = df['SNOW'] * df['TMIN']
    if 'PRCP' in df.columns and 'TMAX' in df.columns:
        df['PRCP_TMAX'] = df['PRCP'] * df['TMAX']
    if 'PRCP' in df.columns and 'AWND' in df.columns:
        df['PRCP_AWND'] = df['PRCP'] * df['AWND']
    if 'SNOW' in df.columns and 'AWND' in df.columns:
        df['SNOW_AWND'] = df['SNOW'] * df['AWND']
    if 'TMAX' in df.columns and 'AWND' in df.columns:
        df['TMAX_AWND'] = df['TMAX'] * df['AWND']
    if 'TMIN' in df.columns and 'AWND' in df.columns:
        df['TMIN_AWND'] = df['TMIN'] * df['AWND']

    # Added Additional features to match the expected input shape (12 outputs)
    additional_features = ['Adverse_Weather', 'SNOW_TMIN', 'PRCP_TMAX', 'PRCP_AWND', 'SNOW_AWND', 'TMAX_AWND', 'TMIN_AWND']
    for feature in additional_features:
        if feature not in df.columns:
            df[feature] = 0  # Replace missing additonal features with 0

    # Apply log transformation to skewed features (reduce skewness)
    df['TMAX'] = np.log1p(df['TMAX'])
    df['TMIN'] = np.log1p(df['TMIN'])
    df['PRCP'] = np.log1p(df['PRCP'])
    df['SNOW'] = np.log1p(df['SNOW'])
    df['AWND'] = np.log1p(df['AWND'])

    # Ensure the feature names and order match those used during training
    # Ordered: TMAX,TMIN,PRCP,SNOW,AWND,Adverse,SNOW_TMIN,PRCP_TMAX,PRCP_AWND,SNOW_AWND,TMAX_AWND,TMIN_AWND
    all_features = required_features + additional_features
    df = df[all_features]

    # Prints Station Ids with correct datatypes
    #print(f"Historical data for stations {station_ids[:stations_with_data]}")
    return df

# Function to make predictions based on users zipcode
def predict_forecast(zipcode):
    lat, lon, city, state = get_lat_lon(zipcode)
    # Error Handling
    if lat is None or lon is None:
        return {"error": "Unable to fetch geolocation data for the provided zipcode"}

    historical_data = fetch_historical_data(lat, lon)
    if historical_data.empty:
        return {"error": "No historical data available for the provided location"}

    # Print first few rows of data frame (BEFORE SCALING)
    #print("Historical data before scaling:")
    #print(historical_data.head())

    # Scale the data
    X_scaled = scaler.transform(historical_data)

    # Ensure the scaled data is in the correct format
    #print("Scaled data:")
    #print(X_scaled[:5])

    # Make predictions with scaling
    tmax_pred = tmax_model.predict(X_scaled)
    tmin_pred = tmin_model.predict(X_scaled)
    prcp_pred = prcp_model.predict(X_scaled)
    snow_pred = snow_model.predict(X_scaled)
    awnd_pred = awnd_model.predict(X_scaled)

    # Ensure the predictions are in the correct format
    #print("Predictions:")
    #print("TMAX:", tmax_pred[:5])
    #print("TMIN:", tmin_pred[:5])
    #print("PRCP:", prcp_pred[:5])
    #print("SNOW:", snow_pred[:5])
    #print("AWND:", awnd_pred[:5])


    # Convert TMAX and TMIN from Celsius to Fahrenheit (TMIN additional scaling)
    tmax_f = (np.mean(tmax_pred) * 9/5) + 32
    tmin_f = ((np.mean(tmin_pred) * 9/5) + 32)* 0.6

    forecast = {
        'TMAX': float(tmax_f),  # Use mean of TMAX predictions
        'TMIN': float(tmin_f),  # Use mean of TMIN predictions
        'PRCP': float(np.mean(prcp_pred)),  # Use mean of PRCP predicitons
        'SNOW': float(np.mean(snow_pred)),  # Use mean of SNOW predictions  
        'AWND': float(np.mean(awnd_pred)),  # Use mean of AWND predictions
        'city': city,
        'state': state
    }
    return forecast

# Handles GET and POST requests 
@app.route('/', methods=['GET', 'POST'])
# Function to access/execute index.html
def index():
    if request.method == 'POST':
        # Extract Zipcode from form submission
        zipcode = request.form['zipcode']
        # call predict_forecast to get forecast
        forecast = predict_forecast(zipcode)
        # Renders web app using index.html to return predictions to user
        return render_template('index.html', forecast=forecast, zipcode=zipcode)
    # for GET, simply renders empty index.html template on webapp
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)