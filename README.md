# WeatherApp.v1: Personalized Weather Predictions

![WeatherApp Logo](weatherapp_logo.png)

**WeatherApp.v1** is a personalized weather prediction application that provides tailored forecasts to help you plan your day according to your preferences for maximum temperature, minimum temperature, humidity, and other key factors. The app utilizes advanced algorithms that combine historical and real-time data to calculate a personalized weather score, ensuring you make informed decisions for a more enjoyable experience.

## Features

- **User Preferences**: Input your preferred maximum temperature, minimum temperature, and humidity to customize your weather predictions.

- **Model Training**: Train the model to adapt to your specific preferences, ensuring accurate and personalized forecasts.

- **Personalized Scores**: Receive personalized weather scores that reflect how well the weather aligns with your preferences.

- **Hourly Ratings**: Get hourly ratings throughout the day based on your preferences to help you plan activities effectively.

## How to Use

1. **Input Preferences**: Launch the WeatherApp and input your desired maximum temperature, minimum temperature, and humidity.

2. **Train the Model**: Train the model to learn your preferences and provide accurate predictions based on historical and real-time data.

3. **Receive Predictions**: Receive personalized weather scores along with a message indicating whether the weather matches your preferences.

4. **Hourly Ratings Plot**: For predictions, you'll also receive an hourly ratings plot that visualizes the predicted weather scores throughout the day.

## Installation and Usage

1. Clone or download the repository to your local machine.

2. Install the required dependencies by running:

   ```
   pip install -r requirements.txt
   ```

3. Run the `main.py` script:

   ```
   python main.py
   ```

4. The app interface will open in your default web browser, allowing you to input your preferences and receive predictions.

## Technologies Used

- Python
- TensorFlow
- scikit-learn
- pandas
- numpy
- gradio
- requests
- matplotlib

## Acknowledgments

Weather data provided by [Visual Crossing Weather API](https://www.visualcrossing.com/weather/weather-data-services#/). Logo design by [Freepik](https://www.freepik.com).

## License

This project is licensed under the [MIT License](LICENSE).
