import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pickle
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from pathlib import Path

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠ TensorFlow not available. Install with: pip install tensorflow")


class WeatherDataProcessor:
    """Enhanced processor with both XGBoost and Deep Learning options."""
    
    def __init__(self, city_name=None, lat=None, lon=None, cache_days=730):
        self.city_name = city_name
        self.lat = lat
        self.lon = lon
        self.cache_days = cache_days
        self.cache_dir = Path("weather_cache")
        self.model_dir = Path("weather_models")
        
        # Create directories if they don't exist
        self.cache_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
        
        # Get coordinates if city name is provided
        if city_name and (lat is None or lon is None):
            self.lat, self.lon = self._geocode_city(city_name)
    
    def _geocode_city(self, city_name):
        """Convert city name to latitude/longitude using geocoding API."""
        try:
            url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&language=en&format=json"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                lat = result['latitude']
                lon = result['longitude']
                print(f"✓ Found coordinates for {city_name}: ({lat}, {lon})")
                return lat, lon
            else:
                raise ValueError(f"Could not find coordinates for city: {city_name}")
        except Exception as e:
            raise ValueError(f"Geocoding error for {city_name}: {str(e)}")
    
    def _get_cache_path(self):
        """Get the cache file path for this city."""
        city_slug = self.city_name.lower().replace(' ', '_') if self.city_name else f"loc_{self.lat}_{self.lon}"
        return self.cache_dir / f"{city_slug}_weather_cache.csv"
    
    def _get_model_path(self, target_var='temperature_2m', model_type='xgboost'):
        """Get the model file path for this city and target variable."""
        city_slug = self.city_name.lower().replace(' ', '_') if self.city_name else f"loc_{self.lat}_{self.lon}"
        if model_type == 'hybrid':
            return self.model_dir / f"{city_slug}_{target_var}_hybrid_model.pkl"
        elif model_type == 'lstm':
            return self.model_dir / f"{city_slug}_{target_var}_lstm_model.h5"
        else:
            return self.model_dir / f"{city_slug}_{target_var}_model.pkl"
    
    def _is_cache_valid(self):
        """Check if cache exists and is recent enough."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False
        
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=1):
            print(f"⚠ Cache is {cache_age.days} days old, will refresh")
            return False
        
        return True
    
    def download_weather_data(self, force_refresh=False):
        """Download historical weather data from Open-Meteo API."""
        cache_path = self._get_cache_path()
        
        if not force_refresh and self._is_cache_valid():
            print(f"✓ Using cached data from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=['date'])
            return df
        
        print(f"⏳ Downloading weather data for the last {self.cache_days} days...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.cache_days)
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': self.lat,
            'longitude': self.lon,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'hourly': [
                'temperature_2m', 'relative_humidity_2m', 'dew_point_2m',
                'apparent_temperature', 'precipitation', 'rain', 'snowfall',
                'snow_depth', 'weather_code', 'pressure_msl', 'surface_pressure',
                'cloud_cover', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high',
                'visibility', 'wind_speed_10m', 'wind_direction_10m', 'wind_gusts_10m',
                'et0_fao_evapotranspiration', 'vapour_pressure_deficit',
                'soil_temperature_0_to_7cm', 'soil_temperature_7_to_28cm',
                'soil_moisture_0_to_7cm', 'soil_moisture_7_to_28cm'
            ],
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            hourly_data = data['hourly']
            df = pd.DataFrame(hourly_data)
            df['date'] = pd.to_datetime(df['time'])
            df = df.drop('time', axis=1)
            
            df.to_csv(cache_path, index=False)
            print(f"✓ Downloaded {len(df)} hourly records")
            print(f"✓ Saved cache to {cache_path}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download weather data: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing weather data: {str(e)}")
    
    def preprocess_data(self, df, target_var='temperature_2m', lag_hours=[1, 2, 3, 6, 12, 24, 48, 72, 168]):
        """Preprocess data and create comprehensive lag features."""
        print(f"⏳ Preprocessing data...")
        
        df = df.set_index('date').sort_index()
        
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data.")
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encodings
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Lag features
        lag_cols = []
        for lag in lag_hours:
            lag_col = f'{target_var}_lag_{lag}h'
            df[lag_col] = df[target_var].shift(lag)
            lag_cols.append(lag_col)
        
        # Rolling statistics
        windows = [3, 6, 12, 24, 48, 72, 168]
        for window in windows:
            df[f'{target_var}_rolling_mean_{window}h'] = df[target_var].rolling(window=window, min_periods=1).mean()
            df[f'{target_var}_rolling_std_{window}h'] = df[target_var].rolling(window=window, min_periods=1).std()
            df[f'{target_var}_rolling_min_{window}h'] = df[target_var].rolling(window=window, min_periods=1).min()
            df[f'{target_var}_rolling_max_{window}h'] = df[target_var].rolling(window=window, min_periods=1).max()
        
        # Exponential weighted moving averages
        df[f'{target_var}_ewm_12h'] = df[target_var].ewm(span=12, adjust=False).mean()
        df[f'{target_var}_ewm_24h'] = df[target_var].ewm(span=24, adjust=False).mean()
        df[f'{target_var}_ewm_168h'] = df[target_var].ewm(span=168, adjust=False).mean()
        
        # Difference features
        diff_lags = [1, 2, 3, 6, 12, 24, 48, 168]
        for lag in diff_lags:
            df[f'{target_var}_diff_{lag}h'] = df[target_var].diff(lag)
        
        # Percentage changes
        df[f'{target_var}_pct_change_24h'] = df[target_var].pct_change(24)
        df[f'{target_var}_pct_change_168h'] = df[target_var].pct_change(168)
        
        # Interaction features
        if 'relative_humidity_2m' in df.columns:
            df['temp_humidity_interaction'] = df[target_var] * df['relative_humidity_2m']
        if 'wind_speed_10m' in df.columns:
            df['temp_wind_interaction'] = df[target_var] * df['wind_speed_10m']
        if 'pressure_msl' in df.columns:
            df['temp_pressure_interaction'] = df[target_var] * df['pressure_msl']
        
        # Lagged weather features
        weather_features = ['relative_humidity_2m', 'wind_speed_10m', 'pressure_msl', 
                           'cloud_cover', 'precipitation']
        for feature in weather_features:
            if feature in df.columns:
                for lag in [1, 3, 6, 24]:
                    df[f'{feature}_lag_{lag}h'] = df[feature].shift(lag)
        
        critical_cols = [target_var] + lag_cols
        initial_len = len(df)
        df = df.dropna(subset=critical_cols)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            print(f"⚠ Dropped {dropped} rows due to missing target/lag values")
        
        if len(df) == 0:
            raise ValueError("No data remaining after preprocessing.")
        
        # Handle missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
                if df[col].isna().any():
                    median_val = df[col].median()
                    fill_val = median_val if not pd.isna(median_val) else 0
                    df[col] = df[col].fillna(fill_val)
        
        print(f"✓ Preprocessed {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def build_lstm_model(self, sequence_length, n_features, n_outputs=1):
        """Build advanced LSTM model with attention mechanism."""
        if not DEEP_LEARNING_AVAILABLE:
            raise RuntimeError("TensorFlow not available. Install with: pip install tensorflow")
        
        model = keras.Sequential([
            # First LSTM layer with return sequences for stacking
            layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, n_features),
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Second LSTM layer
            layers.LSTM(64, return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            
            # Attention layer (simplified)
            layers.LSTM(32, return_sequences=False,
                       kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dense(n_outputs)
        ])
        
        # Use custom learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.001,
            decay_steps=1000,
            decay_rate=0.96
        )
        
        optimizer = keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_sequences(self, df, target_var, sequence_length=168):
        """Prepare sequences for LSTM training."""
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_var in feature_cols:
            feature_cols.remove(target_var)
        
        # Remove features with NaN
        valid_features = [col for col in feature_cols if not df[col].isna().any()]
        
        X_data = df[valid_features].values
        y_data = df[target_var].values
        
        # Create sequences
        X_sequences = []
        y_targets = []
        
        for i in range(len(X_data) - sequence_length):
            X_sequences.append(X_data[i:i+sequence_length])
            y_targets.append(y_data[i+sequence_length])
        
        return np.array(X_sequences), np.array(y_targets), valid_features
    
    def train_hybrid_model(self, df, target_var='temperature_2m', n_splits=10, 
                          xgb_params=None, use_deep_learning=True):
        """Train hybrid ensemble: XGBoost + LSTM."""
        print(f"⏳ Training Hybrid Model (XGBoost + {'LSTM' if use_deep_learning else 'XGBoost only'})...")
        
        target = df[target_var]
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_var in feature_cols:
            feature_cols.remove(target_var)
        
        valid_features = [col for col in feature_cols if not df[col].isna().any()]
        
        X = df[valid_features]
        y = target
        
        print(f"  Features: {len(valid_features)}")
        print(f"  Samples: {len(X)}")
        
        # Train XGBoost
        print("\n  [1/2] Training XGBoost component...")
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'tree_method': 'hist',
                'max_depth': 12,
                'learning_rate': 0.03,
                'n_estimators': 1000,
                'subsample': 0.85,
                'colsample_bytree': 0.85,
                'colsample_bylevel': 0.85,
                'min_child_weight': 2,
                'gamma': 0.05,
                'reg_alpha': 0.5,
                'reg_lambda': 2.0,
                'max_delta_step': 1,
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        
        tscv = TimeSeriesSplit(n_splits=min(n_splits, len(X) // 100))
        xgb_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                        verbose=False)
            
            score = xgb_model.score(X_val, y_val)
            xgb_scores.append(score)
            print(f"    Fold {fold}: R² = {score:.4f}")
        
        print(f"  ✓ XGBoost Average R²: {np.mean(xgb_scores):.4f}")
        
        # Train final XGBoost
        final_xgb = xgb.XGBRegressor(**xgb_params)
        final_xgb.fit(X, y, verbose=False)
        
        lstm_model = None
        lstm_scaler = None
        sequence_features = None
        
        if use_deep_learning and DEEP_LEARNING_AVAILABLE:
            print("\n  [2/2] Training LSTM component...")
            
            sequence_length = 168  # 1 week of hourly data
            
            # Prepare sequences
            X_seq, y_seq, seq_features = self.prepare_sequences(df, target_var, sequence_length)
            
            # Scale data for LSTM
            lstm_scaler = StandardScaler()
            X_seq_scaled = lstm_scaler.fit_transform(X_seq.reshape(-1, X_seq.shape[-1]))
            X_seq_scaled = X_seq_scaled.reshape(X_seq.shape)
            
            # Train/validation split
            split_idx = int(len(X_seq_scaled) * 0.9)
            X_train_seq = X_seq_scaled[:split_idx]
            X_val_seq = X_seq_scaled[split_idx:]
            y_train_seq = y_seq[:split_idx]
            y_val_seq = y_seq[split_idx:]
            
            # Build and train LSTM
            lstm_model = self.build_lstm_model(sequence_length, X_seq.shape[-1])
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
            ]
            
            print("    Training LSTM (this may take several minutes)...")
            history = lstm_model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=100,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            val_loss = min(history.history['val_loss'])
            print(f"  ✓ LSTM Validation Loss: {val_loss:.4f}")
            
            sequence_features = seq_features
        
        # Save hybrid model
        model_path = self._get_model_path(target_var, 'hybrid')
        model_data = {
            'xgb_model': final_xgb,
            'feature_cols': valid_features,
            'target_var': target_var,
            'xgb_score': np.mean(xgb_scores),
            'has_lstm': lstm_model is not None,
            'lstm_scaler': lstm_scaler,
            'sequence_features': sequence_features,
            'sequence_length': 168 if lstm_model else None,
            'trained_at': datetime.now().isoformat()
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        if lstm_model is not None:
            lstm_path = self._get_model_path(target_var, 'lstm')
            lstm_model.save(lstm_path)
            print(f"✓ Saved LSTM model to {lstm_path}")
        
        print(f"✓ Saved hybrid model to {model_path}")
        
        return final_xgb, lstm_model, valid_features, np.mean(xgb_scores)
    
    def run_full_pipeline(self, target_var='temperature_2m', use_deep_learning=True,
                         lag_hours=[1, 2, 3, 6, 12, 24, 48, 72, 168],
                         xgb_params=None, force_refresh=False):
        """Run complete pipeline with hybrid model option."""
        print(f"\n{'='*60}")
        print(f"Advanced Weather Model Training Pipeline")
        print(f"City: {self.city_name or f'({self.lat}, {self.lon})'}")
        print(f"Target: {target_var}")
        print(f"Mode: {'Hybrid (XGBoost + LSTM)' if use_deep_learning and DEEP_LEARNING_AVAILABLE else 'XGBoost Only'}")
        print(f"{'='*60}\n")
        
        if use_deep_learning and not DEEP_LEARNING_AVAILABLE:
            print("⚠ TensorFlow not available. Falling back to XGBoost only.")
            use_deep_learning = False
        
        try:
            df = self.download_weather_data(force_refresh=force_refresh)
            df_processed = self.preprocess_data(df, target_var=target_var, lag_hours=lag_hours)
            
            xgb_model, lstm_model, features, score = self.train_hybrid_model(
                df_processed,
                target_var=target_var,
                use_deep_learning=use_deep_learning,
                xgb_params=xgb_params
            )
            
            print(f"\n{'='*60}")
            print(f"✓ Pipeline completed successfully!")
            print(f"  Model Type: {'Hybrid (XGBoost + LSTM)' if lstm_model else 'XGBoost'}")
            print(f"  XGBoost R² score: {score:.4f}")
            print(f"  Training samples: {len(df_processed)}")
            print(f"  Features: {len(features)}")
            print(f"  Cache: {self._get_cache_path()}")
            print(f"  Model: {self._get_model_path(target_var, 'hybrid')}")
            print(f"{'='*60}\n")
            
            return {
                'xgb_model': xgb_model,
                'lstm_model': lstm_model,
                'features': features,
                'score': score,
                'data': df_processed
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train advanced weather models')
    parser.add_argument('--city', type=str, help='City name')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    parser.add_argument('--target', type=str, default='temperature_2m', help='Target variable')
    parser.add_argument('--force', action='store_true', help='Force refresh cache')
    parser.add_argument('--no-dl', action='store_true', help='Disable deep learning')
    
    args = parser.parse_args()
    
    if not args.city and (args.lat is None or args.lon is None):
        parser.error("Must provide either --city or both --lat and --lon")
    
    processor = WeatherDataProcessor(
        city_name=args.city,
        lat=args.lat,
        lon=args.lon,
        cache_days=args.days
    )
    
    processor.run_full_pipeline(
        target_var=args.target,
        use_deep_learning=not args.no_dl,
        force_refresh=args.force
    )


if __name__ == '__main__':
    main()