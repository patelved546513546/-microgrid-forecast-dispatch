Predictive Energy Management for a Solar + EV Charging Microgrid
Forecast net‑load (P10/P50/P90) with ML
Optimize a battery to cut peak and cost
Live Streamlit app with savings and uncertainty
How to run

pip install -r requirements.txt
python src/train_quantile.py
streamlit run src/streamlit_app.py
Data

Ausgrid “2012–2013 Solar home electricity data v2.csv” (put in data/ausgrid/)
Weather via Open‑Meteo (cached; synthetic fallback)
Results

Baseline RMSE 20.75 → Model RMSE 2.03 (≈90% better), MAPE ≈ 2.51%
P10–P90 coverage ≈ 76% (target ~80% with calibration)
Repo structure

src/: loaders, features, training, battery optimizer, app
models/: saved quantile models and predictions
data/: ausgrid CSV, weather_cache (ignored)