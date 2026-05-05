from fastapi import FastAPI
import mysql.connector
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta, date
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib
import os
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import pickle

app = FastAPI()

# --- MySQL config ---
MYSQL_HOST = "localhost"
MYSQL_USER = "root"
MYSQL_PASSWORD = ""
MYSQL_DB = "Dayao"

def calculate_classification_metrics(y_true, y_pred):
    """Convert regression predictions to binary classification and calculate metrics"""
    try:
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "kappa": None
            }
        
        # Use median as threshold
        threshold = np.median(y_true)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        
        accuracy = round(float(accuracy_score(y_true_binary, y_pred_binary)), 4)
        precision = round(float(precision_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
        recall = round(float(recall_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
        f1 = round(float(f1_score(y_true_binary, y_pred_binary, zero_division=0)), 4)
        kappa = round(float(cohen_kappa_score(y_true_binary, y_pred_binary)), 4)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "kappa": kappa
        }
    except Exception as e:
        print(f"⚠ Classification metrics error: {e}")
        return {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
            "kappa": None
        }

def get_mysql_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )

# ============================================================================
# LOCATION FORECASTING - Uses saved KMeans model (expensive clustering)
# ============================================================================
KMEANS_FILE = "kmeans_model.pkl"
kmeans_model = None
trained_data = None
location_rows = None
cluster_variances = None
last_training_count = 0

def load_kmeans_model():
    """Load saved KMeans model - this is expensive to train, so we cache it"""
    global kmeans_model, trained_data, location_rows, cluster_variances, last_training_count
    if os.path.exists(KMEANS_FILE):
        saved = joblib.load(KMEANS_FILE)
        kmeans_model = saved.get("model")
        trained_data = saved.get("trained_data")
        location_rows = saved.get("location_rows")
        cluster_variances = saved.get("cluster_variances")
        last_training_count = len(location_rows) if location_rows else 0
        print(f"✓ Loaded KMeans model with {last_training_count} locations")

load_kmeans_model()

def train_kmeans():
    """Train KMeans clustering model - only when needed"""
    global kmeans_model, trained_data, location_rows, cluster_variances, last_training_count

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT province_id, city_id, barangay_id, COUNT(*) AS demand_30d
        FROM addresses
        WHERE patient_id IS NOT NULL
          AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
        GROUP BY province_id, city_id, barangay_id
    """
    cursor.execute(query)
    location_rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not location_rows:
        return False

    trained_data = np.array([[row["demand_30d"]] for row in location_rows])
    num_clusters = 3 if len(trained_data) >= 3 else len(trained_data)
    kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_model.fit(trained_data)

    centroids = kmeans_model.cluster_centers_
    labels = kmeans_model.labels_

    cluster_variances = {}
    for c in range(num_clusters):
        points = trained_data[labels == c]
        cluster_variances[c] = float(np.mean(np.linalg.norm(points - centroids[c], axis=1))) if len(points) > 1 else 0.0001

    joblib.dump({
        "model": kmeans_model,
        "trained_data": trained_data,
        "location_rows": location_rows,
        "cluster_variances": cluster_variances
    }, KMEANS_FILE)

    last_training_count = len(location_rows)
    print(f"✓ Trained KMeans with {last_training_count} locations, {num_clusters} clusters")
    return True

@app.get("/train/location")
def train_location():
    """Manually trigger KMeans retraining"""
    if train_kmeans():
        return {
            "status": "trained and saved", 
            "rows": len(trained_data), 
            "clusters": kmeans_model.n_clusters
        }
    return {"error": "No data to train on"}

@app.get("/forecastlocation")
def forecast_location():
    """
    Location demand forecast using KMeans clustering.
    Auto-retrains only when new location data is detected.
    """
    global kmeans_model, trained_data, location_rows, cluster_variances, last_training_count

    # Check if we need to retrain (new locations added)
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM addresses 
        WHERE created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY) 
        AND patient_id IS NOT NULL
    """)
    total_rows = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    # Auto-retrain if data changed or model doesn't exist
    if total_rows != last_training_count or kmeans_model is None:
        print(f"🔄 Retraining KMeans: {last_training_count} -> {total_rows} locations")
        train_kmeans()

    if kmeans_model is None:
        return {"error": "Model not trained"}

    # Generate forecasts using the (possibly updated) model
    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_
    results = []
    today = datetime.today()
    actuals_all, predicted_all = [], []

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)

    if location_rows is None or len(location_rows) == 0:
        print("⚠ No location_rows available")
        conn.close()
        return {
            "clusters": [],
            "forecast_metrics": {
                "MAPE": None,
                "RMSE": None,
                "R2": None,
                "model_status": "no_data",
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1_score": None,
                "kappa": None
            }
        }

    for idx, row in enumerate(location_rows):
        province_id = row["province_id"]
        city_id = row["city_id"]
        barangay_id = row["barangay_id"]
        cluster_id = int(labels[idx])

        cursor.execute("""
            SELECT DATE(created_at) AS day, COUNT(*) AS daily_demand
            FROM addresses
            WHERE patient_id IS NOT NULL
              AND province_id=%s AND city_id=%s AND barangay_id=%s
              AND created_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            GROUP BY DATE(created_at)
            ORDER BY DATE(created_at)
        """, (province_id, city_id, barangay_id))
        daily_data = cursor.fetchall()

        day_map = {r['day'].strftime('%Y-%m-%d'): r['daily_demand'] for r in daily_data}
        daily_series = np.array([
            day_map.get((today - timedelta(days=29-i)).strftime('%Y-%m-%d'), 0) 
            for i in range(30)
        ], dtype=float)

        # Linear regression forecast (fast, always fresh)
        X = np.arange(len(daily_series)).reshape(-1, 1)
        y = daily_series
        model = LinearRegression()
        model.fit(X, y)
        future_X = np.arange(len(daily_series), len(daily_series)+7).reshape(-1, 1)
        forecast_values = [max(0, int(round(val))) for val in model.predict(future_X)]

        # Collect predictions for metrics
        predictions = model.predict(X).flatten()
        actuals_all.extend(y)
        predicted_all.extend(predictions)

        distance = float(np.linalg.norm([sum(y)] - centroids[cluster_id]))
        variance = cluster_variances[cluster_id] or 0.0001
        error_margin = min(distance / variance, 1.0)

        forecast_dict = {
            (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): forecast_values[i] 
            for i in range(7)
        }

        results.append({
            "province_id": province_id,
            "city_id": city_id,
            "barangay_id": barangay_id,
            "demand_30d": int(sum(y)),
            "cluster": cluster_id,
            "distance_from_centroid": round(distance, 4),
            "cluster_variance": round(variance, 4),
            "error_margin": round(error_margin, 4),
            "forecast_next_7_days": forecast_dict
        })

    cursor.close()
    conn.close()

    # Calculate metrics
    mape = rmse = r2 = None
    classification_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "kappa": None
    }
    
    if len(actuals_all) > 1 and len(predicted_all) > 1:
        try:
            actuals_arr = np.array(actuals_all, dtype=float)
            predicted_arr = np.array(predicted_all, dtype=float)
            
            print(f"📊 Location metrics calculated on {len(actuals_arr)} data points")
            
            # MAPE with zero protection
            mask = actuals_arr != 0
            if np.sum(mask) > 0:
                mape = round(float(mean_absolute_percentage_error(actuals_arr[mask], predicted_arr[mask])), 2)
            
            # RMSE - calculate manually
            mse = mean_squared_error(actuals_arr, predicted_arr)
            rmse = round(float(np.sqrt(mse)), 2)
            r2 = round(float(r2_score(actuals_arr, predicted_arr)), 4)
            classification_metrics = calculate_classification_metrics(actuals_arr, predicted_arr)
        except Exception as e:
            print(f"⚠ Location metrics error: {e}")
            import traceback
            traceback.print_exc()
            mape = rmse = r2 = None
    else:
        print(f"⚠ Insufficient data for metrics: actuals={len(actuals_all)}, predicted={len(predicted_all)}")

    return {
        "clusters": results, 
        "forecast_metrics": {
            "MAPE": mape, 
            "RMSE": rmse, 
            "R2": r2,
            "model_status": "retrained" if total_rows != last_training_count else "cached",
            **classification_metrics
        }
    }


# ============================================================================
# WAITLIST FORECASTING - Always uses fresh data (time-series changes daily)
# ============================================================================
@app.get("/forecastwaitlist")
def forecast_waitlist(clinic_id: str = None):
    """
    Waitlist forecast using moving averages and day-of-week patterns.
    Always trains on fresh data for best accuracy.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get ALL historical data
    if clinic_id:
        cursor.execute("""
            SELECT DATE(w.requested_at_date) AS day, COUNT(*) AS waitlist_count
            FROM waitlist w
            JOIN patients p ON w.patient_id = p.patient_id
            WHERE p.clinic_id=%s
            GROUP BY DATE(w.requested_at_date)
            ORDER BY DATE(w.requested_at_date)
        """, (clinic_id,))
    else:
        cursor.execute("""
            SELECT DATE(requested_at_date) AS day, COUNT(*) AS waitlist_count
            FROM waitlist
            WHERE status='waiting'
            GROUP BY DATE(requested_at_date)
            ORDER BY DATE(requested_at_date)
        """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows:
        return {"error": "No waitlist data found"}

    # Prepare time series
    df = pd.DataFrame(rows)
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)
    
    all_days = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(all_days, fill_value=0)
    series = df['waitlist_count'].values
    
    if len(series) < 7:
        avg_value = max(1, int(round(np.mean(series))))
        today = datetime.today()
        forecast_dict = {
            (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): avg_value
            for i in range(7)
        }
        return {
            "forecast_metrics": {"note": "Insufficient data"},
            "forecast_next_7_days": forecast_dict
        }
    
    # Calculate statistics
    non_zero_days = np.count_nonzero(series)
    zero_days = len(series) - non_zero_days
    std_dev = np.std(series)
    mean_val = np.mean(series)
    cv = (std_dev / mean_val * 100) if mean_val > 0 else 0
    
    # Moving averages (weighted toward recent)
    if len(series) >= 14:
        ma_7 = np.mean(series[-7:])
        ma_14 = np.mean(series[-14:])
        ma_30 = np.mean(series[-30:]) if len(series) >= 30 else ma_14
    else:
        ma_7 = ma_14 = ma_30 = mean_val
    
    base_forecast = ma_7 * 0.5 + ma_14 * 0.3 + ma_30 * 0.2
    
    # Trend analysis
    X = np.arange(len(series)).reshape(-1, 1)
    y = series
    lr_model = LinearRegression()
    lr_model.fit(X, y)
    trend_slope = lr_model.coef_[0]
    
    # Day-of-week patterns
    df_dow = pd.DataFrame({
        'count': series,
        'dow': [(df.index.min() + timedelta(days=i)).weekday() for i in range(len(series))]
    })
    
    dow_multipliers = {}
    for dow in range(7):
        dow_values = df_dow[df_dow['dow'] == dow]['count']
        if len(dow_values) >= 4:
            dow_avg = dow_values.mean()
            dow_multipliers[dow] = dow_avg / mean_val if mean_val > 0 else 1.0
        else:
            dow_multipliers[dow] = 1.0
    
    # Generate forecast
    forecast_values = []
    today = datetime.today()
    
    for i in range(7):
        future_date = today + timedelta(days=i+1)
        future_dow = future_date.weekday()
        
        forecast = base_forecast
        forecast += trend_slope * (len(series) + i + 1) * 0.3
        forecast *= dow_multipliers.get(future_dow, 1.0)
        
        if std_dev > 0:
            variation = np.random.normal(0, std_dev * 0.2)
            forecast += variation
        
        final_forecast = max(1, int(round(forecast))) if non_zero_days > 0 else 0
        forecast_values.append(final_forecast)
    
    # Calculate metrics
    predicted = np.convolve(series, np.ones(7)/7, mode='same')
    mask = series > 0
    
    if np.sum(mask) > 10:
        actual_nonzero = series[mask]
        predicted_nonzero = predicted[mask]
        mape = float(np.mean(np.abs((actual_nonzero - predicted_nonzero) / actual_nonzero)) * 100)
        rmse = float(np.sqrt(np.mean((series - predicted)**2)))
        r2 = float(r2_score(series, predicted))
    else:
        mape = float(np.mean(np.abs((series - predicted) / (series + 0.01))) * 100)
        rmse = float(np.sqrt(np.mean((series - predicted)**2)))
        r2 = 0.0
    
    classification_metrics = calculate_classification_metrics(series, predicted)
    
    forecast_dict = {
        (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): forecast_values[i] 
        for i in range(7)
    }
    
    return {
        "forecast_metrics": {
            "MAPE": round(mape, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4),
            "data_points": len(series),
            "non_zero_days": int(non_zero_days),
            "mean": round(mean_val, 2),
            "std_dev": round(std_dev, 2),
            "cv_percent": round(cv, 2),
            "trend_slope": round(trend_slope, 4),
            "forecast_method": "moving_average_with_dow_pattern",
            "clinic_id": clinic_id if clinic_id else "all",
            "model_status": "trained_fresh",
            **classification_metrics
        }, 
        "forecast_next_7_days": forecast_dict,
        "data_quality": {
            "status": "good" if cv < 100 and non_zero_days > len(series) * 0.5 else "sparse",
            "recommendation": "Data suitable for forecasting" if cv < 150 else "Consider collecting more consistent data"
        }
    }


# ============================================================================
# REVENUE FORECASTING - Always uses fresh data (time-series changes daily)
# ============================================================================
@app.get("/forecastrevenue")
def forecast_revenue(clinic_id: str = None):
    """
    Revenue forecast using log-transform, exponential smoothing, and DOW patterns.
    Always trains on fresh data for best accuracy.
    """
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    if clinic_id:
        cursor.execute("""
            SELECT DATE(p.paid_at_date) AS day, SUM(p.amount) AS daily_revenue
            FROM payments p
            WHERE p.clinic_id=%s AND p.paid_at_date IS NOT NULL
            GROUP BY DATE(p.paid_at_date)
            ORDER BY DATE(p.paid_at_date)
        """, (clinic_id,))
    else:
        cursor.execute("""
            SELECT DATE(paid_at_date) AS day, SUM(amount) AS daily_revenue
            FROM payments
            WHERE paid_at_date IS NOT NULL
            GROUP BY DATE(paid_at_date)
            ORDER BY DATE(paid_at_date)
        """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    if not rows:
        return {"error": "No revenue data found"}
    
    # Prepare DataFrame
    df = pd.DataFrame(rows)
    df['day'] = pd.to_datetime(df['day'])
    df['daily_revenue'] = df['daily_revenue'].astype(float)
    df.set_index('day', inplace=True)
    
    all_days = pd.date_range(df.index.min(), df.index.max(), freq='D')
    df = df.reindex(all_days, fill_value=0.0)
    series = df['daily_revenue'].astype(float).values
    
    today = datetime.today()
    
    if len(series) < 14:
        avg_val = float(np.mean(series))
        forecast_dict = {
            (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): round(avg_val, 2)
            for i in range(7)
        }
        return {
            "forecast_metrics": {"note": "Insufficient data for modeling"},
            "forecast_next_7_days": forecast_dict
        }
    
    if np.all(series == 0):
        forecast_dict = {
            (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): 0.0
            for i in range(7)
        }
        return {
            "forecast_metrics": {"note": "No revenue data available"},
            "forecast_next_7_days": forecast_dict
        }
    
    # Log-transform for spike reduction, keep zeros
    series_log = np.log1p(series)
    X = np.arange(len(series_log)).reshape(-1, 1)
    ltm_model = LinearRegression()
    ltm_model.fit(X, series_log)
    trend_slope = float(ltm_model.coef_[0])
    
    if len(series_log) >= 21:
        try:
            exp_model = ExponentialSmoothing(
                series_log,
                trend="add",
                seasonal="add",
                seasonal_periods=7,
                initialization_method="estimated"
            )
            exp_fit = exp_model.fit(optimized=True)
            exp_fitted_log = exp_fit.fittedvalues
            exp_forecast_log = exp_fit.forecast(7)
        except Exception as e:
            print(f"⚠ Revenue smoothing fallback: {e}")
            exp_fitted_log = np.full_like(series_log, np.mean(series_log))
            exp_forecast_log = np.full(7, np.mean(series_log))
    else:
        exp_fitted_log = np.full_like(series_log, np.mean(series_log))
        exp_forecast_log = np.full(7, np.mean(series_log))
    
    # Day-of-week multipliers using original scale
    df_dow = pd.DataFrame({
        'revenue': series,
        'dow': [(df.index.min() + timedelta(days=i)).weekday() for i in range(len(series))]
    })
    mean_val = float(series.mean())
    dow_multipliers = {}
    for dow in range(7):
        dow_vals = df_dow[df_dow['dow'] == dow]['revenue']
        if len(dow_vals) > 0 and mean_val > 0:
            dow_multipliers[dow] = float(dow_vals.mean() / mean_val)
        else:
            dow_multipliers[dow] = 1.0
    
    # Generate forecast
    base_forecast_log = 0.5 * (ltm_model.predict(np.arange(len(series_log), len(series_log) + 7).reshape(-1, 1)) + exp_forecast_log)
    forecast_values = []
    for i in range(7):
        forecast = np.expm1(base_forecast_log[i])
        future_dow = (today + timedelta(days=i+1)).weekday()
        forecast *= dow_multipliers.get(future_dow, 1.0)
        forecast_values.append(round(max(0.0, forecast), 2))
    
    # In-sample fitted values for metrics
    in_sample_log = 0.5 * (ltm_model.predict(X) + exp_fitted_log)
    fitted = np.expm1(in_sample_log)
    
    mask = series > 0
    if np.sum(mask) > 0:
        mape = round(float(mean_absolute_percentage_error(series[mask], fitted[mask])) * 100, 2)
    else:
        mape = None
    rmse = round(float(np.sqrt(mean_squared_error(series, fitted))), 2)
    r2 = round(float(r2_score(series, fitted)), 4)
    classification_metrics = calculate_classification_metrics(series, fitted)
    
    forecast_dict = {
        (today + timedelta(days=i+1)).strftime("%Y-%m-%d"): forecast_values[i]
        for i in range(7)
    }
    
    return {
        "forecast_metrics": {
            "MAPE": mape,
            "RMSE": rmse,
            "R2": r2,
            "data_points": len(series),
            "non_zero_days": int(np.count_nonzero(series)),
            "mean_daily": round(mean_val, 2),
            "trend_slope": round(trend_slope, 4),
            "total_forecast_7d": round(sum(forecast_values), 2),
            "forecast_method": "log_exp_smoothing_dow",
            "clinic_id": clinic_id if clinic_id else "all",
            "model_status": "trained_fresh",
            **classification_metrics
        },
        "forecast_next_7_days": forecast_dict,
        "data_quality": {
            "status": "good" if np.std(series) / mean_val < 1.5 else "moderate",
            "recommendation": "Data suitable for forecasting" if np.count_nonzero(series) / len(series) > 0.6 else "Consider more consistent revenue data",
            "trend_direction": "increasing" if trend_slope > 0 else "decreasing" if trend_slope < 0 else "stable"
        }
    }
## ============================================================================ 
# TREATMENT FORECASTING - CNN MODEL (All Historical Data, Auto-Retrain with Metrics)
# ============================================================================
CNN_MODEL_FILE = "cnn_treatment_model.h5"
SCALER_FILE = "treatment_scaler.pkl"
cnn_model = None
treatment_scaler = None
treatment_names = []
last_treatment_count = 0

def load_cnn_model():
    """Load saved CNN model and scaler"""
    global cnn_model, treatment_scaler, treatment_names, last_treatment_count
    if os.path.exists(CNN_MODEL_FILE) and os.path.exists(SCALER_FILE):
        try:
            cnn_model = keras.models.load_model(CNN_MODEL_FILE, compile=False)
            cnn_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=[keras.metrics.MeanAbsoluteError()]
            )
            with open(SCALER_FILE, 'rb') as f:
                saved = pickle.load(f)
                treatment_scaler = saved.get("scaler")
                treatment_names = saved.get("treatment_names", [])
                last_treatment_count = saved.get("last_count", 0)
            print(f"✓ Loaded CNN model with {len(treatment_names)} treatments")
        except Exception as e:
            print(f"⚠ Failed to load CNN model: {e}")

load_cnn_model()

def prepare_sequences(data, lookback=30):
    """Create sequences for time series prediction"""
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def build_cnn_model(input_shape, num_treatments):
    """Build CNN model architecture"""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(50, activation='relu'),
        layers.Dense(num_treatments, activation='linear')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=[keras.metrics.MeanAbsoluteError()]
    )
    return model

def train_cnn_treatment():
    """Train CNN model on ALL historical treatment data"""
    global cnn_model, treatment_scaler, treatment_names, last_treatment_count

    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Fetch ALL completed treatments (no time limit)
    cursor.execute("""
        SELECT treatment_name, DATE(treatment_date) AS day, COUNT(*) AS daily_count
        FROM patient_treatments
        WHERE status='completed'
        GROUP BY treatment_name, DATE(treatment_date)
        ORDER BY treatment_name, DATE(treatment_date)
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Need at least 40 data points for meaningful training
    if not rows or len(rows) < 40:
        print(f"⚠ Insufficient data: {len(rows) if rows else 0} rows (need 40+)")
        return False

    df = pd.DataFrame(rows)
    df['day'] = pd.to_datetime(df['day'])
    df['daily_count'] = df['daily_count'].astype(float)
    treatment_names[:] = df['treatment_name'].unique().tolist()

    # Build daily matrix across entire date range
    date_range = pd.date_range(df['day'].min(), df['day'].max())
    matrix = []
    for t in treatment_names:
        t_df = df[df['treatment_name'] == t].set_index('day')
        series = pd.Series(index=date_range, dtype=float)
        series.update(t_df['daily_count'])
        series = series.fillna(0.0).values
        matrix.append(series)
    matrix = np.array(matrix).T

    print(f"📊 Training data: {len(matrix)} days, {len(treatment_names)} treatments")

    # Normalize data
    treatment_scaler = StandardScaler()
    normalized = treatment_scaler.fit_transform(matrix)

    # Prepare sequences with 30-day lookback
    lookback = 30
    X, y = prepare_sequences(normalized, lookback)
    
    if len(X) < 10:
        print(f"⚠ Insufficient sequences: {len(X)} (need 10+)")
        return False
    
    X = X.reshape((X.shape[0], X.shape[1], len(treatment_names)))

    # Build and train model
    cnn_model = build_cnn_model((lookback, len(treatment_names)), len(treatment_names))

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = cnn_model.fit(
        X, y,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # Save model and scaler
    cnn_model.save(CNN_MODEL_FILE)
    with open(SCALER_FILE, 'wb') as f:
        pickle.dump({
            "scaler": treatment_scaler,
            "treatment_names": treatment_names,
            "last_count": len(rows)
        }, f)

    last_treatment_count = len(rows)
    print(f"✓ CNN trained: {len(treatment_names)} treatments, {len(X)} sequences")
    print(f"✓ Final val_loss: {history.history['val_loss'][-1]:.4f}")
    
    return True

@app.get("/train/treatment")
def train_treatment():
    """Manually trigger model training"""
    if train_cnn_treatment():
        return {
            "status": "trained and saved",
            "treatment_types": len(treatment_names),
            "treatments": treatment_names,
            "data_points": last_treatment_count
        }
    return {"error": "Insufficient data (need 40+ treatment records)"}

@app.get("/forecasttreatment")
def forecast_treatment(clinic_id: str = None):
    """Generate 7-day treatment forecast with performance metrics"""
    global cnn_model, treatment_scaler, treatment_names, last_treatment_count

    # Check if retraining is needed
    conn = get_mysql_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM patient_treatments
        WHERE status='completed'
    """)
    total_rows = cursor.fetchone()[0]
    cursor.close()
    conn.close()

    retrain_status = "cached"
    if total_rows != last_treatment_count or cnn_model is None:
        print(f"🔄 Retraining: {total_rows} vs {last_treatment_count} records")
        if not train_cnn_treatment():
            return {"error": "Insufficient data for training (need 40+ treatment records)"}
        retrain_status = "retrained"

    # --- Build input sequence (last 30 days) ---
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    query = """
        SELECT treatment_name, DATE(treatment_date) AS day, COUNT(*) AS daily_count
        FROM patient_treatments
        WHERE status='completed' AND treatment_date >= DATE_SUB(NOW(), INTERVAL 30 DAY)
    """
    if clinic_id:
        query += f" AND clinic_id='{clinic_id}'"
    query += " GROUP BY treatment_name, DATE(treatment_date) ORDER BY day"
    
    cursor.execute(query)
    rows_30 = cursor.fetchall()
    cursor.close()
    conn.close()

    if not rows_30:
        return {"error": "No treatment data in last 30 days"}

    # Prepare 30-day input matrix
    df_30 = pd.DataFrame(rows_30)
    df_30['day'] = pd.to_datetime(df_30['day'])
    df_30['daily_count'] = df_30['daily_count'].astype(float)
    date_range_30 = pd.date_range(end=datetime.today(), periods=30)
    
    matrix_30 = []
    for t in treatment_names:
        t_df = df_30[df_30['treatment_name'] == t].set_index('day')
        series = pd.Series(index=date_range_30, dtype=float)
        series.update(t_df['daily_count'])
        series = series.fillna(0.0).values
        matrix_30.append(series)
    matrix_30 = np.array(matrix_30).T
    
    normalized_30 = treatment_scaler.transform(matrix_30)
    X_input = normalized_30.reshape((1, 30, len(treatment_names)))

    # --- Generate 7-day forecast ---
    forecast_results = []
    current_seq = X_input.copy()
    
    for i in range(7):
        pred = cnn_model.predict(current_seq, verbose=0)
        pred_denorm = treatment_scaler.inverse_transform(pred)[0]
        
        forecast_results.append({
            "date": (datetime.today() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "predictions": {
                treatment_names[j]: float(round(max(0, pred_denorm[j]), 2))
                for j in range(len(treatment_names))
            }
        })
        
        # Roll sequence and append prediction
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = pred[0]

    # --- Calculate metrics on ALL historical data ---
    conn = get_mysql_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT treatment_name, DATE(treatment_date) AS day, COUNT(*) AS daily_count
        FROM patient_treatments
        WHERE status='completed'
        GROUP BY treatment_name, DATE(treatment_date)
        ORDER BY day
    """)
    rows_all = cursor.fetchall()
    cursor.close()
    conn.close()

    mse = mae = rmse = mape = None
    validation_days = 0

    if rows_all and len(rows_all) >= 40:
        df_all = pd.DataFrame(rows_all)
        df_all['day'] = pd.to_datetime(df_all['day'])
        df_all['daily_count'] = df_all['daily_count'].astype(float)
        date_range_all = pd.date_range(df_all['day'].min(), df_all['day'].max())
        
        # Build full historical matrix
        matrix_all = []
        for t in treatment_names:
            t_df = df_all[df_all['treatment_name'] == t].set_index('day')
            series = pd.Series(index=date_range_all, dtype=float)
            series.update(t_df['daily_count'])
            series = series.fillna(0.0).values
            matrix_all.append(series)
        matrix_all = np.array(matrix_all).T
        
        validation_days = len(matrix_all)
        
        # Only calculate metrics if we have enough data
        if len(matrix_all) > 30:
            normalized_all = treatment_scaler.transform(matrix_all)
            X_val, y_val = prepare_sequences(normalized_all, 30)
            
            if len(X_val) > 0:
                X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], len(treatment_names)))
                y_pred = cnn_model.predict(X_val, verbose=0)
                
                # Denormalize for metric calculation
                y_val_denorm = treatment_scaler.inverse_transform(y_val)
                y_pred_denorm = treatment_scaler.inverse_transform(y_pred)
                
                # Calculate metrics
                mse = round(float(np.mean((y_val_denorm - y_pred_denorm)**2)), 4)
                mae = round(float(np.mean(np.abs(y_val_denorm - y_pred_denorm))), 4)
                rmse = round(float(np.sqrt(np.mean((y_val_denorm - y_pred_denorm)**2))), 4)
                
                # MAPE with protection against division by zero
                mask = y_val_denorm > 0.1  # Only include non-zero actuals
                if np.any(mask):
                    mape = round(float(np.mean(np.abs((y_val_denorm[mask] - y_pred_denorm[mask]) / y_val_denorm[mask])) * 100), 2)
                else:
                    mape = None
                
                print(f"📈 Metrics calculated on {len(X_val)} validation sequences")

    # Calculate 7-day totals
    totals = {
        t: round(sum([f["predictions"][t] for f in forecast_results]), 2)
        for t in treatment_names
    }
    
    # Calculate classification metrics on validation data
    classification_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "kappa": None
    }
    
    if 'y_val_denorm' in locals() and 'y_pred_denorm' in locals() and len(y_val_denorm) > 0:
        try:
            # Flatten predictions and actuals
            y_true_flat = y_val_denorm.flatten()
            y_pred_flat = y_pred_denorm.flatten()
            classification_metrics = calculate_classification_metrics(y_true_flat, y_pred_flat)
        except Exception as e:
            classification_metrics["error"] = str(e)

    return {
        "forecast_metrics": {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "treatment_types": len(treatment_names),
            "validation_days": validation_days,
            "validation_sequences": len(X_val) if 'X_val' in locals() and len(X_val) > 0 else 0,
            "model_status": retrain_status,
            "clinic_id": clinic_id if clinic_id else "all",
            **classification_metrics
        },
        "forecast_next_7_days": forecast_results,
        "treatment_totals_7d": totals,
        "treatments": treatment_names
    }

# Update the root endpoint to include treatment forecasting
@app.get("/")
def root():
    return {
        "service": "AI Forecast API",
        "version": "2.1",
        "endpoints": {
            "/forecastlocation": "Location demand (KMeans clustering + linear regression, cached)",
            "/forecastwaitlist": "Waitlist prediction (moving average + DOW pattern, fresh)",
            "/forecastrevenue": "Revenue prediction (log-trend + exponential smoothing + DOW, fresh)",
            "/forecasttreatment": "Treatment prediction (CNN on historical treatments, cached)",
            "/train/location": "Manually retrain location model",
            "/train/treatment": "Manually retrain treatment model"
        },
        "strategy": {
            "location": "Cached KMeans clustering model with fresh linear regression forecasts",
            "waitlist": "Fresh moving-average forecast with daily seasonality",
            "revenue": "Fresh log-trend + exponential smoothing forecast with DOW adjustments",
            "treatment": "Cached CNN model that retrains when new completed treatments arrive"
        }
    }