import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os
import calendar
from streamlit_folium import folium_static
import folium
from folium.plugins import HeatMap

# Set page config
st.set_page_config(page_title="Chennai Rainfall Prediction", page_icon="🌧️", layout="wide")

# Custom CSS to improve the app's appearance
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #FFFFFF, #F0F8FF);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1E90FF;
    }
    .stButton>button {
        background-color: #1E90FF;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        background-color: #F0F8FF;
        color: #1E90FF;
        border-radius: 5px;
    }
    .stSelectbox>div>div>select {
        background-color: #F0F8FF;
        color: #1E90FF;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess data
# more than 40k+ records in csv
@st.cache_data
def load_and_preprocess_data():
    file_path = 'dataset/Preprocessed_Data.csv'
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("Error: File was not found.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("Error: The file is empty.")
        st.stop()
    except pd.errors.ParserError:
        st.error("Error: Unable to parse the CSV file. Please ensure it is a valid CSV file.")
        st.stop()

    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
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
    data['season'] = data['month'].map(lambda x: 'Winter' if x in [12, 1, 2] else 
                                       'Spring' if x in [3, 4, 5] else 
                                       'Summer' if x in [6, 7, 8] else 'Fall')
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['is_weekend'] = data['day_of_week'].isin([5, 6])

    # Add sine and cosine features for cyclical time variables
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    data['day_sin'] = np.sin(2 * np.pi * data['day'] / 31)
    data['day_cos'] = np.cos(2 * np.pi * data['day'] / 31)

    # Encode categorical variables
    label_encoder_district = LabelEncoder()
    label_encoder_station = LabelEncoder()
    data['dist_encoded'] = label_encoder_district.fit_transform(data['District'])
    data['station_encoded'] = label_encoder_station.fit_transform(data['Station'])

    return data, label_encoder_district, label_encoder_station

# Train the model
# utilizing deep learning model and Keras API
@st.cache_resource
def train_model(X_train, y_train):
    input_shape = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)), 
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, 
                        epochs=10,
                        batch_size=32, 
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0)

    return model, history

# Predict rainfall likelihood
def predict_rain(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler):
    try:
        user_date = pd.to_datetime(user_input_date)
    except Exception as e:
        st.error(f"Error parsing the input date: {e}")
        return []

    # Generate necessary features for prediction
    julian_date = user_date.to_julian_date()
    day = user_date.day
    month = user_date.month
    year = user_date.year
    day_of_week = user_date.dayofweek
    is_weekend = day_of_week in [5, 6]
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_sin = np.sin(2 * np.pi * day / 31)
    day_cos = np.cos(2 * np.pi * day / 31)

    predictions = []
    for dist in data['District'].unique():
        dist_encoded = label_encoder_district.transform([dist])[0]
        for station in data[data['District'] == dist]['Station'].unique():
            station_encoded = label_encoder_station.transform([station])[0]
            input_data = np.array([[julian_date, dist_encoded, station_encoded, day, month, year,
                                    day_of_week, is_weekend, month_sin, month_cos, day_sin, day_cos]])
            input_data_scaled = scaler.transform(input_data)
            rain_prob = np.clip(model.predict(input_data_scaled)[0][0], 0, 1)
            predictions.append((dist, station, rain_prob * 100))  

    return predictions

# Save model and scaler
def save_model_and_scaler(model, scaler):
    model.save('rainfall_model.h5')
    joblib.dump(scaler, 'scaler.joblib')

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = tf.keras.models.load_model('rainfall_model.h5')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Create a map of Chennai
def create_chennai_map(data):
    chennai_center = [13.0827, 80.2707]
    m = folium.Map(location=chennai_center, zoom_start=10)

    # Create markers for each station
    for _, row in data.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=5,
            popup=f"Station: {row['Station']}<br>Avg Rainfall: {row['Rainfall']:.2f} mm",
            color='blue',
            fill=True,
            fillColor='blue'
        ).add_to(m)

    return m

# Streamlit app
def main():
    st.title("🌧️ Chennai Rainfall Prediction App")

    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predictions", "Data Exploration", "Model Performance","About"])

    # Load and preprocess data
    data, label_encoder_district, label_encoder_station = load_and_preprocess_data()

    if page == "Home":
        st.write("Welcome to the Chennai Rainfall Prediction App! This application uses machine learning to predict the likelihood of rainfall in different areas of Chennai.")
        st.write("Use the sidebar to navigate through different sections of the app.")

        st.subheader("Quick Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(data):,}")
        with col2:
            st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
        with col3:
            st.metric("Rainy Days", f"{data['rain'].sum():,} ({data['rain'].mean()*100:.2f}%)")
        with col4:
            st.metric("Avg. Rainfall", f"{data['Rainfall'].mean():.2f} mm")
        # fixes
        st.subheader("Recent Weather Trends")
        recent_data = data.sort_values('Date').tail(30)
        fig = px.line(recent_data, x='Date', y='Rainfall', title='Rainfall in the Last 30 Days')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Rainfall Distribution by Season")
        season_data = data.groupby('season')['Rainfall'].mean().reset_index()
        fig = px.bar(season_data, x='season', y='Rainfall', title='Average Rainfall by Season',
                     color='season', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Predictions":
        st.header("Rainfall Predictions")

        relevant_features = ['julian_date', 'dist_encoded', 'station_encoded', 'day', 'month', 'year',
                             'day_of_week', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        X = data[relevant_features]
        y = data['rain'].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if not os.path.exists('rainfall_model.h5') or not os.path.exists('scaler.joblib'):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            model, history = train_model(X_train, y_train)
            save_model_and_scaler(model, scaler)
        else:
            model, scaler = load_model_and_scaler()

        col1, col2 = st.columns(2)
        with col1:
            user_input_date = st.date_input("Select a date to predict rainfall:",
                                            min_value=data['Date'].min().date(),
                                            max_value=data['Date'].max().date() + timedelta(days=365*500))  # Allow predictions up to 500 years in the future
        with col2:
            selected_district = st.selectbox("Select a district:", options=['All'] + list(data['District'].unique()))

        if st.button("Predict", key='predict_button'):
            if user_input_date is None:
                st.error("Please select a valid date.")
                st.error("The date should be within the range of available data.")
            else:
                predictions = predict_rain(model, user_input_date, data, label_encoder_district, label_encoder_station, scaler)

                if predictions:
                    st.subheader(f"Predictions for {user_input_date.strftime('%d-%m-%Y')}")
                    
                    if selected_district != 'All':
                        predictions = [p for p in predictions if p[0] == selected_district]
                    
                    st.write("### Top 5 Locations Most Likely to Receive Rain:")
                    top_5 = predictions[:5]
                    for i, (district, station, rain_prob) in enumerate(top_5, 1):
                        st.write(f"{i}. **District:** {district}, **Station:** {station}, **Rain Probability:** {rain_prob:.2f}%")

                    
                    st.subheader("All Predictions")
                    df_predictions = pd.DataFrame(predictions, columns=['District', 'Station', 'Rain Probability (%)'])
                    
                    fig = px.scatter(df_predictions, x='District', y='Rain Probability (%)', 
                                        hover_data=['Station'],
                                        title='Rainfall Probability by District')
                    st.plotly_chart(fig, use_container_width=True)


                    pivot_df = df_predictions.pivot(index='District', columns='Station', values='Rain Probability (%)')
                    fig = px.imshow(pivot_df, title='Heatmap of Rainfall Probabilities',
                                    labels=dict(x="Station", y="District", color="Rain Probability (%)"))
                    st.plotly_chart(fig, use_container_width=True)

    elif page == "Data Exploration":
        st.header("Data Exploration")

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Rainfall distribution
        st.subheader("Rainfall Distribution")
        fig = px.histogram(data, x='Rainfall', nbins=50, title='Distribution of Rainfall')
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall by district
        st.subheader("Average Rainfall by District")
        district_rainfall = data.groupby('District')['Rainfall'].mean().sort_values(ascending=False)
        fig = px.bar(district_rainfall, title='Average Rainfall by District')
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall by month
        st.subheader("Average Rainfall by Month")
        monthly_rainfall = data.groupby('month')['Rainfall'].mean()
        fig = px.line(monthly_rainfall, x=monthly_rainfall.index, y=monthly_rainfall.values, 
                      labels={'x': 'Month', 'y': 'Average Rainfall'},
                      title='Average Rainfall by Month')
        fig.update_xaxes(tickmode='array', tickvals=list(range(1, 13)), ticktext=calendar.month_abbr[1:])
        st.plotly_chart(fig, use_container_width=True)

        # Rainfall by day of week
        st.subheader("Average Rainfall by Day of Week")
        dow_rainfall = data.groupby('day_of_week')['Rainfall'].mean()
        fig = px.bar(dow_rainfall, x=dow_rainfall.index, y=dow_rainfall.values,
                     labels={'x': 'Day of Week', 'y': 'Average Rainfall'},
                     title='Average Rainfall by Day of Week')
        fig.update_xaxes(tickmode='array', tickvals=list(range(7)), ticktext=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        corr = data[['Rainfall', 'julian_date', 'day', 'month', 'year', 'day_of_week', 'is_weekend']].corr()
        fig = px.imshow(corr, title='Correlation Heatmap')
        st.plotly_chart(fig, use_container_width=True)

    elif page == "Model Performance":
        st.header("Model Performance")

        relevant_features = ['julian_date', 'dist_encoded', 'station_encoded', 'day', 'month', 'year',
                             'day_of_week', 'is_weekend', 'month_sin', 'month_cos', 'day_sin', 'day_cos']
        X = data[relevant_features]
        y = data['rain'].astype(int)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        if os.path.exists('rainfall_model.h5'):
            model, _ = load_model_and_scaler()
        else:
            model, history = train_model(X_train, y_train)
            save_model_and_scaler(model, scaler)

        try:
            test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

            st.subheader("Model Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Test Accuracy", f"{test_accuracy:.2f}")
            col2.metric("Test Precision", f"{test_precision:.2f}")
            col3.metric("Test Recall", f"{test_recall:.2f}")
            col4.metric("Test Loss", f"{test_loss:.2f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            df_report = pd.DataFrame(report).transpose()
            st.table(df_report)

            # ROC Curve
            st.subheader("ROC Curve")
            y_pred_proba = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC Curve (AUC = {roc_auc:.2f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random Classifier'))
            fig.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                              xaxis_title='False Positive Rate',
                              yaxis_title='True Positive Rate')
            st.plotly_chart(fig, use_container_width=True)

            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)
            feature_names = relevant_features
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)

            fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred while evaluating the model: {str(e)}")
            st.info("Please ensure that the model is properly trained and saved.")

    elif page == "About":
        st.header("About This App")
        st.write("""
        This Chennai Rainfall Prediction App is designed to provide insights into rainfall patterns in Chennai and predict the likelihood of rain for specific dates and locations.

        Key Features:
        1. Data Exploration: Analyze historical rainfall data through various visualizations.
        2. Predictions: Get rainfall predictions for any date within the dataset range and up to 500 years in the future.
        3. Model Performance: Evaluate the machine learning model's performance through various metrics and visualizations.
        4. Rainfall Map: Visualize rainfall patterns across Chennai using an interactive map.

        The app uses a deep learning model trained on historical data to make predictions. It considers factors such as date, location, season, and day of the week to estimate the probability of rainfall.

        Data Source: The data used in this app is sourced from historical weather information for Chennai.

        Note: Predictions are based on historical patterns and may not account for all factors affecting weather. Always refer to official weather forecasts for critical decision-making.

        Credits: John Paul , Frontend Developer
                 Param Varsha , Streamlit Developer
                 Karan S , Data Researcher and Analyst
        Last Updated: 
        Version: 1.0.0
        """)

        # Add a feedback section
        st.subheader("Feedback")
        feedback = st.text_area("We'd love to hear your thoughts on the app. Please leave your feedback below:")
        if st.button("Submit Feedback"):
            st.success("Thank you for your feedback!")

        # Add version information
        st.sidebar.info("App Version: 1.0.0")

    # Add a footer
    st.markdown("---")
    st.markdown("© 2024 Chennai Rainfall Prediction App. All rights reserved.")

if __name__ == "__main__":
    main()
