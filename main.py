import streamlit as st
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from datetime import datetime

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    file_path = 'dataset/chennai-all-stations-daily-rainfall.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please ensure it's in the correct location.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error(f"Error: The file '{file_path}' is empty.")
        st.stop()
    except pd.errors.ParserError:
        st.error(f"Error: Unable to parse '{file_path}'. Please ensure it's a valid CSV file.")
        st.stop()
    
    data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')
    data['julian_date'] = data['Date'].map(pd.Timestamp.to_julian_date)
    data['day'] = data['Date'].dt.day
    data['month'] = data['Date'].dt.month
    data['year'] = data['Date'].dt.year
    data['rain'] = data['Rainfall'] > 0
    
    label_encoder_district = LabelEncoder()
    label_encoder_station = LabelEncoder()
    data['dist_encoded'] = label_encoder_district.fit_transform(data['District'])
    data['station_encoded'] = label_encoder_station.fit_transform(data['Station'])
    
    return data, label_encoder_district, label_encoder_station

# Train the model
@st.cache_resource
def train_model(X_train, y_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu', input_shape=[X_train.shape[1]]),
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

# Predict rainfall likelihood
def predict_rain(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler):
    julian_date = pd.to_datetime(user_input_date).to_julian_date()
    day = pd.to_datetime(user_input_date).day
    month = pd.to_datetime(user_input_date).month
    year = pd.to_datetime(user_input_date).year
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
            predictions.append((dist, station, rain_prob * 100))  # Convert to percentage

    predictions.sort(key=lambda x: x[2], reverse=True)
    return predictions

# Streamlit app
def main():
    st.title("Chennai Rainfall Prediction App")
    
    # Load and preprocess data
    data, label_encoder_district, label_encoder_station = load_and_preprocess_data()
    
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
    st.write(f"Test Accuracy: {test_accuracy:.2f}")
    st.write(f"Test Precision: {test_precision:.2f}")
    st.write(f"Test Recall: {test_recall:.2f}")
    
    # Plot training history
    st.subheader("Training History")
    history_df = pd.DataFrame(history.history)
    st.line_chart(history_df[['accuracy', 'val_accuracy']])
    
    # User input
    user_input_date = st.date_input("Select a date to predict rainfall:")
    
    if st.button("Predict Rainfall"):
        predictions = predict_rain(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler)
        
        st.subheader(f"Rainfall Predictions for {user_input_date}")
        
        # Display top 5 predictions
        st.write("Top 5 locations most likely to receive rain:")
        for i, (district, station, probability) in enumerate(predictions[:5], 1):
            st.write(f"{i}. District: {district}, Station: {station}, Probability: {probability:.2f}%")
        
        # Plot all predictions
        st.subheader("All Predictions")
        df_predictions = pd.DataFrame(predictions, columns=['District', 'Station', 'Probability'])
        st.bar_chart(df_predictions.set_index('Station')['Probability'])

if __name__ == "__main__":
    main()