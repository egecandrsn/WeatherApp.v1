import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
from datetime import datetime, timedelta
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import json
import requests
import gradio as gr
import os.path
import matplotlib.pyplot as plt
import tempfile

data=pd.read_csv("weatherdatafinal.csv")

def add_daytime_column(data):
    data['sunrise'] = pd.to_datetime(data['sunrise'])
    data['sunset'] = pd.to_datetime(data['sunset'])
    
    data['daytime'] = (data['sunset'] - data['sunrise']).dt.total_seconds() / 3600.0
    
    return data

data=add_daytime_column(data)

data = data.drop(columns=['name','datetime', 'severerisk', 'conditions', 'description', 'icon', 'stations','snow','snowdepth','sunrise','sunset','precip'])
data['preciptype'] = data['preciptype'].fillna(0)
data['preciptype'] = data['preciptype'].replace({'rain': 1, 'rain,snow': 2, 'snow': 3, 'rain,freezingrain,snow':3})
data['windgust'] = data['windgust'].fillna(data['windgust'].median())
data['sealevelpressure'] = data['sealevelpressure'].fillna(data['sealevelpressure'].median())
data['pressure'] = data['sealevelpressure']
data=data.drop("sealevelpressure", axis=1)
feature_names = list(data.columns)
def train_model(ideal_max_temp, ideal_min_temp, ideal_humidity):

    
    ideal_weights = {
        'tempmax': 6,
        'tempmin': 6,
        'temp': 6,
        'humidity': 2,
        'windspeed': 3,
        'windgust': 1.5,
        'cloudcover': 3,
        'daytime': 1,
        'precipprob': 1.5,
        'visibility': 1,
        'stability': 1
    }
    
    def normalize(value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    
    for idx, row in data.iterrows():
        tempmax_score = 1-normalize(abs(ideal_max_temp - row['tempmax']), min(data['tempmax']), max(data['tempmax']))
        tempmin_score = 1-normalize(abs(ideal_min_temp - row['tempmin']), min(data['tempmin']), max(data['tempmin']))
        temp_score = 1-normalize(abs(((ideal_max_temp + ideal_min_temp) / 2) - row['temp']), min(data['temp']), max(data['temp']))
        stability_score = 1-normalize(abs(row['tempmax'] - row['tempmin']), min(data['tempmin']), max(data['tempmax']))
        humidity_score = 1-normalize(abs(ideal_humidity - row['humidity']), min(data['humidity']), max(data['humidity']))
        windspeed_score = 1-normalize(row['windspeed'], min(data['windspeed']), max(data['windspeed']))
        windgust_score = 1-normalize(row['windgust'], min(data['windgust']), max(data['windgust']))
        cloudcover_score = 1-normalize(row['cloudcover'], min(data['cloudcover']), max(data['cloudcover']))
        daytime_score = normalize(row['daytime'], min(data['daytime']), max(data['daytime']))
        precipprob_score = 1-normalize(row['precipprob'], min(data['precipprob']), max(data['precipprob']))
        visibility_score = normalize(row['visibility'], min(data['visibility']), max(data['visibility']))
        
        scores = [
            tempmax_score * ideal_weights['tempmax'],
            tempmin_score * ideal_weights['tempmin'],
            temp_score * ideal_weights['temp'],
            humidity_score * ideal_weights['humidity'],
            windspeed_score * ideal_weights['windspeed'],
            windgust_score * ideal_weights['windgust'],
            cloudcover_score * ideal_weights['cloudcover'],
            daytime_score * ideal_weights['daytime'],
            precipprob_score * ideal_weights['precipprob'],
            visibility_score * ideal_weights['visibility'],
            stability_score * ideal_weights['stability']
        ]
    
        daily_score = np.mean(scores)
        data.loc[idx, 'daily_score'] = daily_score


    
    scaler = MinMaxScaler(feature_range=(0, 95))
    
    scaled_scores = scaler.fit_transform(data[['daily_score']])
    
    data['daily_score'] = scaled_scores

    X_train, X_test, y_train, y_test = train_test_split(data.drop('daily_score', axis=1), data['daily_score'], test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mse', 'mae', 'msle'])

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=20, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2, callbacks=[early_stop])

    model.save('trainedmodel.h5')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

        return "Model trained based on your preferences."



def predict_weather(Location, Day):

    model = tf.keras.models.load_model('trainedmodel.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)


    prediction_day = Day.strip().lower()
    if prediction_day == "yesterday":
        day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    elif prediction_day == "today":
        day = datetime.now().strftime("%Y-%m-%d")
    elif prediction_day == "tomorrow":
        day = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        print("Invalid prediction day. Defaulting to today.")
        day = datetime.now().strftime("%Y-%m-%d")

    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{Location}/{day}/{day}?unitGroup=metric&include=hours&key=DEP8HRPCBQRZH5UKZWH32YC2F&contentType=json"
    response = requests.get(url)
    urldata= response.json()
    def add_daytime_column(urldata):
        urldata['days'][0]['sunrise'] = pd.to_datetime(urldata['days'][0]['sunrise'])
        urldata['days'][0]['sunset'] = pd.to_datetime(urldata['days'][0]['sunset'])
        urldata['days'][0]['daytime'] = (urldata['days'][0]['sunset'] - urldata['days'][0]['sunrise']).total_seconds() / 3600.0
        return urldata
        
    urldata=add_daytime_column(urldata)
    
    def preprocess_preciptype(urldata):
        preciptype_dict = {'rain': 1, 'rain,snow': 2, 'snow': 3, 'rain,freezingrain,snow': 4,'None':0}

        for day in urldata['days']:
            if day.get('preciptype') is not None:
                preciptype_str = day['preciptype'][0] 
                preciptype_code = preciptype_dict.get(preciptype_str, 0) 
                day['preciptype'] = preciptype_code 
            else:
                day['preciptype'] = 0 

        return urldata

    def replace_nan_with_median(urldata, data):
        for col in ['solarradiation', 'solarenergy', 'uvindex']:
            urldata['days'][0][col] = urldata['days'][0][col] or np.nan

        for col in ['solarradiation', 'solarenergy', 'uvindex']:
            if np.isnan(urldata['days'][0][col]):
                urldata['days'][0][col] = data[col].median()

        return urldata

    urldata=replace_nan_with_median(urldata,data)
    urldata=preprocess_preciptype(urldata)

    def mean(data, key):
        values = [hour[key] for hour in data]
        return sum(values) / len(values)
    
    hours_data = urldata["days"][0]["hours"][6:24]
    day_data = urldata['days'][0]
    
    new_data = {
        'tempmax': [day_data['tempmax']],
        'tempmin': [day_data['tempmin']],
        'temp': [mean(hours_data, "temp")],
        'feelslikemax': [day_data['feelslikemax']],
        'feelslikemin': [day_data['feelslikemin']],
        'feelslike': [mean(hours_data, "feelslike")],
        'dew': [mean(hours_data, "dew")],
        'humidity': [mean(hours_data, "humidity")],
        'precipprob': [mean(hours_data, "precipprob")],
        'precipcover': [day_data['precipcover']],
        'preciptype': [day_data['preciptype']],
        'windgust': [mean(hours_data, "windgust")],
        'windspeed': [mean(hours_data, "windspeed")],
        'winddir': [day_data['winddir']],
        'pressure': [mean(hours_data, "pressure")],
        'cloudcover': [mean(hours_data, "cloudcover")],
        'visibility': [mean(hours_data, "visibility")],
        'solarradiation': [day_data['solarradiation']],
        'solarenergy': [day_data['solarenergy']],
        'uvindex': [day_data['uvindex']],
        'moonphase': [day_data['moonphase']],
        'daytime': [day_data['daytime']]}

    input_data = pd.DataFrame(new_data)
    input_data = input_data[feature_names]
    input_data = scaler.transform(input_data)
    input_data = input_data.reshape(1, -1)


    predictions = model.predict(input_data)
    hourly_scores = []

    new_data_hour = {
    'tempmin': day_data['tempmin'],
    'feelslikemin': day_data['feelslikemin'],
    'precipcover': day_data['precipcover'],
    'moonphase': day_data['moonphase'],
    'daytime': day_data['daytime']
    }

    for hour_data in hours_data:
        new_data_hour.update({
        'tempmax': hour_data['temp'],
        'feelslikemax': hour_data['feelslike'],
        'temp': hour_data['temp'],
        'feelslike': hour_data['feelslike'],
        'dew': hour_data['dew'],
        'humidity': hour_data['humidity'],
        'precipprob': hour_data['precipprob'],
        'preciptype': day_data['preciptype'],
        'windgust': hour_data['windgust'],
        'windspeed': hour_data['windspeed'],
        'winddir': hour_data['winddir'],
        'pressure': hour_data['pressure'],
        'cloudcover': hour_data['cloudcover'],
        'visibility': hour_data['visibility'],
        'solarradiation': hour_data['solarradiation'],
        'solarenergy': hour_data['solarenergy'],
        'uvindex': hour_data['uvindex']


    })

        input_data_hour = pd.DataFrame([new_data_hour])
        input_data_hour = input_data_hour[feature_names]
        input_data_hour = scaler.transform(input_data_hour)
        input_data_hour = input_data_hour.reshape(1, -1)

        predictions_hour = model.predict(input_data_hour)
        hourly_scores.append(predictions_hour[0][0])
    score = predictions[0][0]
    if score >= 80:
        message = "The weather is expected to be great based on your preferences!"
    elif score >= 60:
        message = "The weather is expected to be good based on your preferences."
    else:
        message = "The weather might not be ideal based on your preferences."

    return score, message, hourly_scores

def main():
    mode = gr.inputs.Radio(["Train Model", "Predict Weather"], label="Mode")
    ideal_max_temp = gr.inputs.Slider(minimum=0, maximum=40, step=1, default=28, label="Ideal max temperature (°C)")
    ideal_min_temp = gr.inputs.Slider(minimum=0, maximum=40, step=1, default=20, label="Ideal min temperature (°C)")
    ideal_humidity = gr.inputs.Slider(minimum=40, maximum=100, step=1, default=70, label="Ideal humidity level (%)")
    Location = gr.inputs.Textbox(placeholder="Enter your location (city name)")
    Day = gr.inputs.Radio(choices=["yesterday", "today", "tomorrow"], label="Select day:")

    outputs = [
        gr.outputs.Textbox(label="Training Result"),
        gr.outputs.Textbox(label="Predicted Daily Score"),
        gr.outputs.Textbox(label="Message"),
        gr.outputs.Image(type="filepath", label="Hourly Rating Plot")
    ]

 
    def wrapper(mode, ideal_max_temp, ideal_min_temp, ideal_humidity, Location, Day):
        if mode == "Train Model":
            result = train_model(ideal_max_temp, ideal_min_temp, ideal_humidity)
            return result, None, None, None, None
        else:
            score, message, hourly_scores = predict_weather(Location, Day)
            hours = range(6, 24)
            plt.plot(hours,hourly_scores)
            plt.xlabel('Hour of the Day')
            plt.ylabel('Hourly Rating')
            plt.xticks(range(6, 25, 1))
            plt.xlim(6, 24)
            plt.yticks(range(0, 101, 10))
            plt.ylim(0, 100)
            plt.title('Hourly Ratings Based On Your Preferences')
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                plt.savefig(temp_file.name, format='png')
                img_filepath = temp_file.name
            plt.clf()

            return None, score, message, img_filepath


    

    interface = gr.Interface(
        fn=wrapper,
        inputs=[mode, ideal_max_temp, ideal_min_temp, ideal_humidity, Location, Day],
        outputs=outputs,
        title="Weather Rating",
        description=(
        "🌤️ <b>WeatherApp.v1: Personalized Weather Predictions</b><br>"
        "Designed to provide you with tailored weather forecasts, taking into account your preferences for maximum and minimum temperature, humidity, and other key factors. Our advanced algorithms calculate weather features using historical and real-time data, delivering a personalized weather score to help you plan your day with confidence.<br><br>"
        "<b>How to use:</b><br>"
        "1. 🌡️ Input your preferred maximum temperature, minimum temperature, and humidity.<br>"
        "2. 🔄 Train the model to adapt to your preferences.<br>"
        "3. 🔮 Receive personalized weather scores to better plan your day.<br><br>"
        "Whether you're planning outdoor activities or just want to know how the day will feel, WeatherApp gives you a user-focused forecast for a more enjoyable experience."
        ),
        allow_flagging=False,
        allow_screenshot=False
    )

    interface.launch()

if __name__ == "__main__":
    main()

