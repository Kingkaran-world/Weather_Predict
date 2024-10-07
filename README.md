<p align="center">
  <img src="https://github.com/RJohnPaul/YouNotes/blob/67253a9687c34fc289b2898b3338b49ad80312a1/younotes.png" alt="YouNotes Logo"/>
</p>

# Weatherit

![Chennai Rainfall Prediction](https://img.shields.io/badge/Chennai-Rainfall%20Prediction-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.2.0-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)


A sophisticated web application that predicts rainfall in Chennai using machine learning techniques. This app provides valuable insights into rainfall patterns and offers predictions for specific dates and locations within Chennai.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Model](#model)
- [App Structure](#app-structure)
- [Contributing](#contributing)
- [Credits](#credits)
- [License](#license)

## Features

- **Home**: Quick statistics and recent weather trends
- **Predictions**: Get rainfall predictions for any date (up to 500 years in the future)
- **Data Exploration**: Analyze historical rainfall data through various visualizations
- **Model Performance**: Evaluate the machine learning model's performance
- **About**: Information about the app and feedback submission

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/RJohnPaul/Weather_Predict.git
   cd Weather_Predict
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:

```
streamlit run main.py
```

Navigate to the provided local URL in your web browser to use the app.

## Data

The app uses historical weather data for Chennai. The dataset includes the following features:

- Date
- District
- Station
- Rainfall
- Latitude
- Longitude

## Model

The rainfall prediction model is a deep neural network built with TensorFlow. It uses the following features:

- Julian date
- District (encoded)
- Station (encoded)
- Day, Month, Year
- Day of the week
- Is weekend
- Cyclical encodings of month and day

The model is trained to predict the likelihood of rainfall for a given set of inputs.

## App Structure

The app is divided into several sections:

1. **Home**: Displays quick stats and recent weather trends
2. **Predictions**: Allows users to select a date and district for rainfall predictions
3. **Data Exploration**: Provides various visualizations of the historical data
4. **Model Performance**: Shows the model's accuracy, precision, recall, and other metrics
5. **About**: Gives information about the app and allows users to submit feedback

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

- John Paul: Frontend Developer
- Param Varsha: Streamlit Developer
- Karan S: Data Researcher and Analyst

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

© 2024 Chennai Rainfall Prediction App. All rights reserved.
