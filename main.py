import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime
import xml.etree.ElementTree as ET
import geopandas as gpd
from shapely.geometry import Polygon

# Load and preprocess data from two CSV files
@st.cache_data
def load_and_preprocess_data():
    file_path1 = 'dataset/set1.csv'
    file_path2 = 'dataset/set2.csv'

    try:
        data1 = pd.read_csv(file_path1)
        data2 = pd.read_csv(file_path2)
    except FileNotFoundError:
        st.error(f"Error: One or both files were not found.")
        st.toast("Please check CSV paths and try again.", type="error")
        # The step above will make sure that the Toast message is also being showwcased in the UI
        # The st.stop() function will stop the execution of the code
        st.stop()
    except pd.errors.EmptyDataError:
        st.error(f"Error: One or both files are empty.")
        st.stop()
    except pd.errors.ParserError:
        st.error(f"Error: Unable to parse the CSV files. Please ensure they are valid CSV files.")
        st.stop()

    # Combine the two datasets
    data = pd.concat([data1, data2], ignore_index=True)

    # Handle date parsing with mixed formats
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

    # Check for invalid dates
    if data['Date'].isnull().any():
        num_invalid_dates = data['Date'].isnull().sum()
        st.warning(f"Warning: {num_invalid_dates} rows have invalid dates and will be removed.")
        data = data.dropna(subset=['Date'])

    # Feature Engineering
    data['julian_date'] = data['Date'].map(pd.Timestamp.to_julian_date)
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['rain'] = data['Rainfall'] > 0

    # Encode categorical variables
    label_encoder_district = LabelEncoder()
    label_encoder_station = LabelEncoder()
    data['dist_encoded'] = label_encoder_district.fit_transform(data['District'])
    data['station_encoded'] = label_encoder_station.fit_transform(data['Station'])

    return data, label_encoder_district, label_encoder_station

# Load flood-prone data from KML
@st.cache_data
def load_flood_data(kml_file):
    try:
        tree = ET.parse(kml_file)
        root = tree.getroot()

        # Get the namespace from the KML file
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        flood_data = []

        # Extract flood areas from KML
        for placemark in root.findall('.//kml:Placemark', ns):
            name_element = placemark.find('kml:name', ns)
            name = name_element.text if name_element is not None else "Unnamed"

            coords_element = placemark.find('.//kml:coordinates', ns)
            if coords_element is not None and coords_element.text.strip():
                coords_text = coords_element.text.strip()

                # Parse the coordinates into (longitude, latitude) tuples
                coords = []
                for coord in coords_text.split():
                    lon, lat, _ = map(float, coord.split(','))
                    coords.append((lon, lat))

                # Create a polygon from the coordinates
                polygon = Polygon(coords)
                flood_data.append({'name': name, 'polygon': polygon})

        return flood_data

    except ET.ParseError:
        st.error("Error parsing KML file. Please ensure the KML file is correctly formatted.")
        return []
    except FileNotFoundError:
        st.error("Error: KML file not found.")
        return []

# Check if a station is in a flood-prone area
def is_flood_prone(station_location, flood_data):
    station_point = Point(station_location)

    for flood_area in flood_data:
        flood_polygon = gpd.GeoSeries(flood_area['polygon']).unary_union
        if flood_polygon.contains(station_point):
            return True

    return False

# Train the model
@st.cache_resource
def train_model(X_train, y_train):
    input_shape = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=100,
                        batch_size=32, 
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0)

    return model, history

# Predict rainfall likelihood and flood probability
def predict_rain_and_flood(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler, flood_data):
    try:
        user_date = pd.to_datetime(user_input_date)
    except Exception as e:
        st.error(f"Error parsing the input date: {e}")
        return []

    julian_date = user_date.to_julian_date()
    day = user_date.day
    month = user_date.month
    year = user_date.year
    districts = data['District'].unique()
    predictions = []

    for dist in districts:
        dist_encoded = label_encoder_district.transform([dist])[0]
        stations = data[data['District'] == dist]['Station'].unique()

        for station in stations:
            station_encoded = label_encoder_station.transform([station])[0]
            input_data = np.array([[julian_date, dist_encoded, station_encoded, day, month, year]])
            input_data_scaled = scaler.transform(input_data)
            rain_prob = model.predict(input_data_scaled)[0][0]

            # Apply logistic function to spread out probabilities
            rain_prob = 1 / (1 + np.exp(-10 * (rain_prob - 0.5)))

            # Check flood probability
            station_location = (data.loc[data['Station'] == station, 'Longitude'].values[0],
                                data.loc[data['Station'] == station, 'Latitude'].values[0])

            flood_prob = 100 if is_flood_prone(station_location, flood_data) else 0

            predictions.append((dist, station, rain_prob * 100, flood_prob))

    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions

# Streamlit app
def main():
    st.title("Chennai Rainfall & Flood Prediction App")

    # Load and preprocess data
    data, label_encoder_district, label_encoder_station = load_and_preprocess_data()

    # Load flood data
    flood_data = load_flood_data('dataset/chennai_cflows_200_yr_return_periods.kml')

    # Prepare features and target
    X = data[['julian_date', 'dist_encoded', 'station_encoded', 'day', 'month', 'year']]
    y = data['rain'].astype(int)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model, history = train_model(X_train, y_train)

    # Evaluate the model
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

    st.subheader("Model Performance")
    st.write(f"**Test Accuracy:** {test_accuracy:.2f}")
    st.write(f"**Test Precision:** {test_precision:.2f}")
    st.write(f"**Test Recall:** {test_recall:.2f}")

    # Plot training history
    st.subheader("Training History")
    history_df = pd.DataFrame(history.history)
    st.line_chart(history_df[['accuracy', 'val_accuracy']])

    # User input
    user_input_date = st.date_input("Select a date to predict rainfall and flood:")

    if st.button("Predict"):
        if user_input_date is None:
            st.error("Please select a valid date.")
        else:
            predictions = predict_rain_and_flood(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler, flood_data)

            if predictions:
                st.subheader(f"Predictions for {user_input_date.strftime('%d-%m-%Y')}")
                
                # Display top 5 predictions
                st.write("### Top 5 Locations Most Likely to Receive Rain:")
                top_5 = predictions[:5]
                for i, (district, station, rain_prob, flood_prob) in enumerate(top_5, 1):
                    st.write(f"{i}. **District:** {district}, **Station:** {station}, **Rain Probability:** {rain_prob:.2f}%, **Flood Probability:** {flood_prob:.2f}%")
                
                # Plot all predictions
                st.subheader("All Predictions")
                df_predictions = pd.DataFrame(predictions, columns=['District', 'Station', 'Rain Probability (%)', 'Flood Probability (%)'])
                st.dataframe(df_predictions)

if __name__ == "__main__":
    main()
