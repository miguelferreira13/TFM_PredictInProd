from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return {"greeting": "Hello world"}

@app.get("/predict")
def predict(pickup_datetime, pickup_longitude, pickup_latitude, dropoff_longitude,\
    dropoff_latitude, passenger_count):

    key = '2013-07-06 17:18:00.000000119'
    # pickup_datetime = object(pickup_datetime)
    pickup_longitude = float(pickup_longitude)
    pickup_latitude = float(pickup_latitude)
    dropoff_longitude = float(dropoff_longitude)
    dropoff_latitude = float(dropoff_latitude)
    passenger_count = int(passenger_count)
    
    pickup_datetime = datetime.strptime(pickup_datetime, "%Y-%m-%d %H:%M:%S")
    # localize the user datetime with NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")
    formatted_pickup_datetime
    # return {
    #     'pickup_datetime' : pickup_datetime,
    #     'pickup_longitude' : pickup_longitude,
    #     'pickup_latitude' : pickup_latitude,
    #     'dropoff_longitude' : dropoff_longitude,
    #     'dropoff_latitude' : dropoff_latitude,
    #     'passenger_count' : passenger_count
    # }
    convert_np = np.array([key, formatted_pickup_datetime, pickup_longitude, pickup_latitude\
        ,dropoff_longitude, dropoff_latitude, passenger_count])
    
    loaded_model = joblib.load('model.joblib')
    pred = loaded_model.predict(pd.DataFrame(convert_np.reshape(-1, len(convert_np)),
                   columns=['key', 'pickup_datetime', 'pickup_longitude', 'pickup_latitude'\
                    ,'dropoff_longitude', 'dropoff_latitude', 'passenger_count']))
    return {'prediction': pred[0]}
