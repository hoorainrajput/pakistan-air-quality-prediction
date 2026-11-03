import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LinearRegression

# Load processed data and train the regression model (same features as Practical 10)
df = pd.read_csv("data/air_quality_processed.csv")
X = df.drop(columns=["pm2_5", "aqi_category"])
y = df["pm2_5"]
model = LinearRegression()
model.fit(X, y)

# Load cleaned data for the overview section (it still has the city column;
# the processed file replaced city with one-hot columns)
overview = pd.read_csv("data/pakistan_air_quality_daily_cleaned.csv")


def pm25_to_aqi_category(pm25):
    if pm25 <= 12:
        return "Good"
    if pm25 <= 35.4:
        return "Moderate"
    if pm25 <= 55.4:
        return "Unhealthy for Sensitive Groups"
    if pm25 <= 150.4:
        return "Unhealthy"
    if pm25 <= 250.4:
        return "Very Unhealthy"
    return "Hazardous"


def season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


st.title("Pakistan Air Quality / Smog Prediction")
st.write("Predict PM2.5 from weather, city and month, and flag unhealthy days.")

st.header("Enter conditions")
city = st.selectbox("City", ["Lahore", "Karachi", "Islamabad", "Peshawar"])
month = st.selectbox("Month", list(range(1, 13)))
temperature_max = st.slider("Temperature max (°C)", -5.0, 50.0, 30.0)
temperature_min = st.slider("Temperature min (°C)", -5.0, 50.0, 20.0)
temperature_mean = (temperature_max + temperature_min) / 2
humidity_mean = st.slider("Humidity mean (%)", 0.0, 100.0, 50.0)
precipitation_sum = st.slider("Precipitation sum (mm)", 0.0, 50.0, 0.0)
wind_speed_max = st.slider("Wind speed max (km/h)", 0.0, 100.0, 10.0)
pm10 = st.slider("PM10", 0.0, 600.0, 100.0)

if st.button("Predict"):
    row = {col: 0 for col in X.columns}
    row["temperature_mean"] = temperature_mean
    row["humidity_mean"] = humidity_mean
    row["precipitation_sum"] = precipitation_sum
    row["wind_speed_max"] = wind_speed_max
    row["pm10"] = pm10
    row["temp_range"] = temperature_max - temperature_min
    row["month"] = month
    row[f"city_{city}"] = 1
    row[f"season_{season_from_month(month)}"] = 1
    input_df = pd.DataFrame([row], columns=X.columns)
    prediction = float(model.predict(input_df)[0])
    category = pm25_to_aqi_category(prediction)
    st.subheader(f"Predicted PM2.5: {prediction:.1f}")
    st.write(f"AQI category: {category}")
    if category in ["Unhealthy for Sensitive Groups", "Unhealthy", "Very Unhealthy", "Hazardous"]:
        st.warning("Unhealthy day: limit outdoor activity.")
    else:
        st.success("Air quality is acceptable.")

st.header("Data overview")
st.dataframe(overview[overview["city"] == city].head(10))
city_means = overview.groupby("city")["pm2_5"].mean().sort_values(ascending=False)
fig, ax = plt.subplots()
city_means.plot(kind="bar", ax=ax)
ax.set_ylabel("PM2.5")
ax.set_title("Mean PM2.5 by City")
st.pyplot(fig)
