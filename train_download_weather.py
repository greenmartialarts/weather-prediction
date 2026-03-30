import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pickle
import os
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from pathlib import Path


class WeatherDataProcessor:
    """Handles downloading, preprocessing, and training weather models for specific cities."""
    
    def __init__(self, city_name=None, lat=None, lon=None, cache_days=7300):
        self.city_name = city_name
        self.lat = lat
        self.lon = lon
        self.cache_days = cache_days  # Default to 2 years for better training
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
    
    def _get_model_path(self, target_var='temperature_2m'):
        """Get the model file path for this city and target variable."""
        city_slug = self.city_name.lower().replace(' ', '_') if self.city_name else f"loc_{self.lat}_{self.lon}"
        return self.model_dir / f"{city_slug}_{target_var}_model.pkl"
    
    def _is_cache_valid(self):
        """Check if cache exists and is recent enough."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return False
        
        # Check if cache is older than 1 day
        cache_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if cache_age > timedelta(days=1):
            print(f"⚠ Cache is {cache_age.days} days old, will refresh")
            return False
        
        return True
    
    def download_weather_data(self, force_refresh=False):
        """Download historical weather data from Open-Meteo API."""
        cache_path = self._get_cache_path()
        
        # Check cache validity
        if not force_refresh and self._is_cache_valid():
            print(f"✓ Using cached data from {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=['date'])
            return df
        
        print(f"⏳ Downloading weather data for the last {self.cache_days} days...")
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=self.cache_days)
        
        # Build API URL with comprehensive weather variables
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
            # Make API request
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            hourly_data = data['hourly']
            df = pd.DataFrame(hourly_data)
            df['date'] = pd.to_datetime(df['time'])
            df = df.drop('time', axis=1)
            
            # Save to cache
            df.to_csv(cache_path, index=False)
            print(f"✓ Downloaded {len(df)} hourly records")
            print(f"✓ Saved cache to {cache_path}")
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Failed to download weather data: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Error processing weather data: {str(e)}")
    
    def preprocess_data(self, df, target_var='temperature_2m', lag_hours=[1, 2, 3, 6, 12, 24, 48]):
        """Preprocess data and create lag features with more context."""
        print(f"⏳ Preprocessing data...")
        
        # Set date as index and sort
        df = df.set_index('date').sort_index()
        
        # Check if target variable exists
        if target_var not in df.columns:
            raise ValueError(f"Target variable '{target_var}' not found in data. Available columns: {df.columns.tolist()}")
        
        # Add time-based features for better pattern recognition
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day_of_year'] = df.index.dayofyear
        
        # Add cyclical encoding for hour (captures daily cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Add cyclical encoding for day of year (captures seasonal cycle)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Create lag features for target variable BEFORE handling missing values
        lag_cols = []
        for lag in lag_hours:
            lag_col = f'{target_var}_lag_{lag}h'
            df[lag_col] = df[target_var].shift(lag)
            lag_cols.append(lag_col)
        
        # Create rolling mean features for smoothing
        df[f'{target_var}_rolling_mean_6h'] = df[target_var].rolling(window=6, min_periods=1).mean()
        df[f'{target_var}_rolling_mean_24h'] = df[target_var].rolling(window=24, min_periods=1).mean()
        
        # Create difference features (rate of change)
        df[f'{target_var}_diff_1h'] = df[target_var].diff(1)
        df[f'{target_var}_diff_24h'] = df[target_var].diff(24)
        
        # Only drop rows where target variable or lag features have NaN
        critical_cols = [target_var] + lag_cols
        initial_len = len(df)
        df = df.dropna(subset=critical_cols)
        dropped = initial_len - len(df)
        
        if dropped > 0:
            print(f"⚠ Dropped {dropped} rows due to missing target/lag values")
        
        if len(df) == 0:
            raise ValueError("No data remaining after preprocessing. Dataset too small.")
        
        # Now handle missing values in other columns
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
    
    def train_model(self, df, target_var='temperature_2m', n_splits=5, 
                   xgb_params=None, save_model=True):
        """Train XGBoost model using time series cross-validation."""
        print(f"⏳ Training model for {target_var}...")
        
        # Prepare features and target
        target = df[target_var]
        
        # Select numeric features only, excluding the target
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_var in feature_cols:
            feature_cols.remove(target_var)
        
        # Remove features that have any NaN values (these are columns with no data)
        valid_features = []
        for col in feature_cols:
            if not df[col].isna().any():
                valid_features.append(col)
            else:
                print(f"  Skipping feature '{col}' (contains NaN values)")
        
        feature_cols = valid_features
        X = df[feature_cols]
        y = target
        
        if len(X) == 0:
            raise ValueError("No features available for training")
        
        print(f"  Features: {len(feature_cols)} numeric columns")
        print(f"  Samples: {len(X)} records")
        
        # Adjust n_splits if dataset is too small
        min_samples_per_split = 50
        max_possible_splits = len(X) // min_samples_per_split
        if max_possible_splits < n_splits:
            n_splits = max(2, max_possible_splits)
            print(f"  Adjusted to {n_splits} splits due to dataset size")
        
        # Default XGBoost parameters - optimized for accuracy
        if xgb_params is None:
            xgb_params = {
                'objective': 'reg:squarederror',
                'max_depth': 8,  # Deeper trees for more complex patterns
                'learning_rate': 0.05,  # Lower learning rate for better accuracy
                'n_estimators': 300,  # More trees
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,  # Regularization
                'gamma': 0.1,  # Regularization
                'reg_alpha': 0.1,  # L1 regularization
                'reg_lambda': 1.0,  # L2 regularization
                'random_state': 42
            }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        models = []
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model = xgb.XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            # Evaluate
            score = model.score(X_val, y_val)
            scores.append(score)
            models.append(model)
            
            print(f"  Fold {fold}/{n_splits}: R² = {score:.4f}")
        
        avg_score = np.mean(scores)
        print(f"✓ Average R² score: {avg_score:.4f}")
        
        # Train final model on all data
        print(f"⏳ Training final model on full dataset...")
        final_model = xgb.XGBRegressor(**xgb_params)
        final_model.fit(X, y, verbose=False)
        
        # Save model
        if save_model:
            model_path = self._get_model_path(target_var)
            model_data = {
                'model': final_model,
                'feature_cols': feature_cols,
                'target_var': target_var,
                'avg_score': avg_score,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            print(f"✓ Saved model to {model_path}")
        
        return final_model, feature_cols, avg_score
    
    def run_full_pipeline(self, target_var='temperature_2m', lag_hours=[1, 2, 3, 6, 12, 24, 48],
                         xgb_params=None, force_refresh=False):
        """Run the complete pipeline: download → preprocess → train."""
        print(f"\n{'='*60}")
        print(f"Weather Model Training Pipeline")
        print(f"City: {self.city_name or f'({self.lat}, {self.lon})'}")
        print(f"Target: {target_var}")
        print(f"{'='*60}\n")
        
        try:
            # Step 1: Download data
            df = self.download_weather_data(force_refresh=force_refresh)
            
            # Step 2: Preprocess data
            df_processed = self.preprocess_data(df, target_var=target_var, lag_hours=lag_hours)
            
            # Step 3: Train model
            model, features, score = self.train_model(
                df_processed, 
                target_var=target_var,
                xgb_params=xgb_params
            )
            
            print(f"\n{'='*60}")
            print(f"✓ Pipeline completed successfully!")
            print(f"  Model R² score: {score:.4f}")
            print(f"  Cache: {self._get_cache_path()}")
            print(f"  Model: {self._get_model_path(target_var)}")
            print(f"{'='*60}\n")
            
            return {
                'model': model,
                'features': features,
                'score': score,
                'data': df_processed
            }
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {str(e)}")
            raise


def main():
    """Example usage of the weather data processor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download and train weather prediction models')
    parser.add_argument('--city', type=str, help='City name (e.g., "San Francisco")')
    parser.add_argument('--lat', type=float, help='Latitude')
    parser.add_argument('--lon', type=float, help='Longitude')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data (default: 365)')
    parser.add_argument('--target', type=str, default='temperature_2m', help='Target variable')
    parser.add_argument('--force', action='store_true', help='Force refresh cache')
    
    args = parser.parse_args()
    
    if not args.city and (args.lat is None or args.lon is None):
        parser.error("Must provide either --city or both --lat and --lon")
    
    # Create processor
    processor = WeatherDataProcessor(
        city_name=args.city,
        lat=args.lat,
        lon=args.lon,
        cache_days=args.days
    )
    
    # Run pipeline
    processor.run_full_pipeline(
        target_var=args.target,
        force_refresh=args.force
    )


if __name__ == '__main__':
    main()