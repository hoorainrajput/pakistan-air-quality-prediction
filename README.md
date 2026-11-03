# Pakistan Air Quality / Smog Prediction

Predict PM2.5 and AQI category for Pakistani cities from weather, season and city, and flag unhealthy days.

## Dataset

Daily data for Lahore, Karachi, Islamabad and Peshawar, 2023-01-01 to 2026-11-01 (5,604 rows).
PM2.5/PM10 from the Open-Meteo Air Quality API (CAMS model); weather from the Open-Meteo Archive API.

## Models

- Linear Regression for PM2.5: MAE 8.61, RMSE 13.24, R² 0.80
- k-NN for AQI category: accuracy 0.79, weighted F1 0.78
- Naive Bayes for AQI category: accuracy 0.32, weighted F1 0.41 (weak: features are correlated)
- k-means clustering (k=3, k=4) for pollution regimes: clean / moderate / severe days

## Run locally

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Project structure

- `app.py` — Streamlit app
- `air_quality_project.ipynb` — full pipeline notebook
- `data/` — daily dataset, cleaned dataset, processed features
