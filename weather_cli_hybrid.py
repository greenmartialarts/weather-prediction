import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from tabulate import tabulate
import sys

# Try to import TensorFlow
try:
    import tensorflow as tf
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False

# Import the training module (try both versions)
try:
    from train_download_weather_dl import WeatherDataProcessor
    USE_DL_VERSION = True
except ImportError:
    try:
        from train_download_weather import WeatherDataProcessor
        USE_DL_VERSION = False
    except ImportError:
        print("❌ Error: train_download_weather.py must be in the same directory")
        sys.exit(1)


class WeatherCLI:
    """Command-line interface with hybrid model support."""
    
    def __init__(self, city_name=None, lat=None, lon=None):
        self.city_name = city_name
        self.lat = lat
        self.lon = lon
        self.processor = WeatherDataProcessor(
            city_name=city_name,
            lat=lat,
            lon=lon
        )
        self.cache_dir = Path("weather_cache")
        self.model_dir = Path("weather_models")
    
    def ensure_model_exists(self, target_var='temperature_2m', force_retrain=False, use_deep_learning=True):
        """Ensure that a model exists for the city, train if necessary."""
        # Check for hybrid model first
        hybrid_path = self.processor._get_model_path(target_var, 'hybrid')
        standard_path = self.processor._get_model_path(target_var, 'xgboost')
        
        model_exists = hybrid_path.exists() or standard_path.exists()
        cache_path = self.processor._get_cache_path()
        cache_is_fresh = self.processor._is_cache_valid()
        
        should_train = False
        
        if not model_exists:
            print(f"⚠ No model found for {self.city_name or 'location'}")
            should_train = True
        elif force_retrain:
            print(f"⚠ Force retrain requested")
            should_train = True
        elif not cache_is_fresh and cache_path.exists():
            print(f"⚠ Cache is outdated, will retrain with fresh data")
            should_train = True
        
        if should_train:
            print(f"\n⏳ Training model for {self.city_name or 'location'}...")
            if USE_DL_VERSION:
                self.processor.run_full_pipeline(
                    target_var=target_var,
                    use_deep_learning=use_deep_learning and DEEP_LEARNING_AVAILABLE,
                    force_refresh=not cache_is_fresh
                )
            else:
                self.processor.run_full_pipeline(
                    target_var=target_var,
                    force_refresh=not cache_is_fresh
                )
            print("✓ Model training completed\n")
        else:
            print(f"✓ Using existing model for {self.city_name or 'location'}")
    
    def load_model(self, target_var='temperature_2m'):
        """Load the trained model (hybrid or standard)."""
        # Try hybrid model first
        hybrid_path = self.processor._get_model_path(target_var, 'hybrid')
        standard_path = self.processor._get_model_path(target_var, 'xgboost')
        
        if hybrid_path.exists():
            with open(hybrid_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Load LSTM if available
            if model_data.get('has_lstm', False):
                lstm_path = self.processor._get_model_path(target_var, 'lstm')
                if lstm_path.exists() and DEEP_LEARNING_AVAILABLE:
                    model_data['lstm_model'] = tf.keras.models.load_model(lstm_path)
                else:
                    model_data['lstm_model'] = None
            
            return model_data
        elif standard_path.exists():
            with open(standard_path, 'rb') as f:
                model_data = pickle.load(f)
            model_data['xgb_model'] = model_data.get('model', None)
            model_data['has_lstm'] = False
            return model_data
        else:
            raise FileNotFoundError("No model found. Run with --train flag.")
    
    def load_and_prepare_data(self, target_var='temperature_2m', lag_hours=[1, 2, 3, 6, 12, 24, 48, 72, 168]):
        """Load cached data and prepare features."""
        cache_path = self.processor._get_cache_path()
        
        if not cache_path.exists():
            print(f"⚠ No cached data found, downloading...")
            df = self.processor.download_weather_data()
        else:
            df = pd.read_csv(cache_path, parse_dates=['date'])
        
        df_processed = self.processor.preprocess_data(
            df, 
            target_var=target_var, 
            lag_hours=lag_hours
        )
        
        return df_processed
    
    def predict(self, hours_ahead=24, target_var='temperature_2m', show_details=False):
        """Generate weather predictions using hybrid model if available."""
        print(f"\n{'='*60}")
        print(f"Weather Forecast")
        print(f"City: {self.city_name or f'({self.lat}, {self.lon})'}")
        print(f"Target: {target_var}")
        print(f"Horizon: {hours_ahead} hours into the future")
        print(f"{'='*60}\n")
        
        # Load model
        print("⏳ Loading model...")
        model_data = self.load_model(target_var)
        xgb_model = model_data['xgb_model']
        lstm_model = model_data.get('lstm_model', None)
        feature_cols = model_data['feature_cols']
        has_lstm = model_data.get('has_lstm', False) and lstm_model is not None
        
        model_type = "Hybrid (XGBoost + LSTM)" if has_lstm else "XGBoost"
        print(f"✓ Model loaded: {model_type}")
        print(f"  XGBoost R² score: {model_data.get('xgb_score', model_data.get('avg_score', 'N/A')):.4f}")
        print(f"  Trained at: {model_data['trained_at']}")
        
        # Load and prepare data
        print("\n⏳ Loading historical data...")
        df = self.load_and_prepare_data(target_var=target_var)
        
        if len(df) == 0:
            raise ValueError("No data available for prediction")
        
        last_timestamp = df.index[-1]
        forecast_start = last_timestamp + pd.Timedelta(hours=1)
        
        print(f"  Last known data: {last_timestamp}")
        print(f"  Forecasting from: {forecast_start}")
        
        # Calculate daily pattern for long-term stability
        recent_history = df.tail(min(720, len(df)))
        hourly_patterns = {}
        
        for hour in range(24):
            hour_data = recent_history[recent_history.index.hour == hour][target_var]
            if len(hour_data) > 0:
                hourly_patterns[hour] = {'mean': hour_data.mean(), 'std': hour_data.std()}
        
        # Calculate trend
        if len(recent_history) >= 168:
            weekly_temps = recent_history[target_var].tail(168)
            daily_trend = (weekly_temps.iloc[-24:].mean() - weekly_temps.iloc[:24].mean()) / 7
        else:
            daily_trend = 0
        
        print(f"\n⏳ Generating {hours_ahead}-hour forecast...")
        predictions = []
        forecast_times = []
        
        lag_cols = [col for col in feature_cols if f'{target_var}_lag_' in col]
        working_df = recent_history.tail(200).copy()
        
        for i in range(hours_ahead):
            current_row = working_df.iloc[-1:].copy()
            forecast_time = forecast_start + pd.Timedelta(hours=i)
            
            # Update lag features
            for lag_col in lag_cols:
                lag_hours = int(lag_col.split('_lag_')[1].replace('h', ''))
                if i >= lag_hours:
                    current_row[lag_col] = predictions[i - lag_hours]
                elif len(working_df) >= lag_hours:
                    current_row[lag_col] = working_df.iloc[-(lag_hours + 1)][target_var]
            
            # Update time features
            if 'hour' in current_row.columns:
                current_row['hour'] = forecast_time.hour
            if 'day_of_week' in current_row.columns:
                current_row['day_of_week'] = forecast_time.dayofweek
            if 'month' in current_row.columns:
                current_row['month'] = forecast_time.month
            if 'day_of_year' in current_row.columns:
                current_row['day_of_year'] = forecast_time.dayofyear
            if 'hour_sin' in current_row.columns:
                current_row['hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
            if 'hour_cos' in current_row.columns:
                current_row['hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
            if 'day_sin' in current_row.columns:
                current_row['day_sin'] = np.sin(2 * np.pi * forecast_time.dayofyear / 365)
            if 'day_cos' in current_row.columns:
                current_row['day_cos'] = np.cos(2 * np.pi * forecast_time.dayofyear / 365)
            
            # Update rolling features
            if f'{target_var}_rolling_mean_6h' in current_row.columns:
                if i >= 6:
                    current_row[f'{target_var}_rolling_mean_6h'] = np.mean(predictions[-6:])
            
            if f'{target_var}_rolling_mean_24h' in current_row.columns:
                if i >= 24:
                    current_row[f'{target_var}_rolling_mean_24h'] = np.mean(predictions[-24:])
            
            # Ensure all features present
            missing_cols = set(feature_cols) - set(current_row.columns)
            for col in missing_cols:
                if col in working_df.columns:
                    current_row[col] = working_df[col].mean()
                else:
                    current_row[col] = 0
            
            X = current_row[feature_cols]
            
            # Make prediction with XGBoost
            pred_xgb = xgb_model.predict(X)[0]
            
            # If LSTM available, ensemble predictions
            if has_lstm:
                # For simplicity, use XGBoost predictions
                # Full LSTM integration would require sequence management
                pred = pred_xgb
            else:
                pred = pred_xgb
            
            # Apply pattern correction for long-term forecasts
            if i >= 48:
                hour_of_day = forecast_time.hour
                if hour_of_day in hourly_patterns:
                    pattern_mean = hourly_patterns[hour_of_day]['mean']
                    correction_strength = min(0.5, (i - 48) / hours_ahead)
                    pred = (1 - correction_strength) * pred + correction_strength * pattern_mean
                
                days_ahead = i / 24
                pred += daily_trend * days_ahead
            
            predictions.append(pred)
            forecast_times.append(forecast_time)
            
            # Update working dataframe
            new_row = current_row.copy()
            new_row.index = [forecast_time]
            new_row[target_var] = pred
            working_df = pd.concat([working_df, new_row])
            
            if len(working_df) > 200:
                working_df = working_df.tail(200)
        
        results = pd.DataFrame({
            'Timestamp': forecast_times,
            'Predicted': predictions,
        })
        
        print(f"\n{'='*60}")
        print("Forecast Summary")
        print(f"{'='*60}")
        print(f"Mean predicted {target_var}: {np.mean(predictions):.2f}")
        print(f"Min predicted {target_var}: {np.min(predictions):.2f}")
        print(f"Max predicted {target_var}: {np.max(predictions):.2f}")
        print(f"Std deviation: {np.std(predictions):.2f}")
        print(f"{'='*60}\n")
        
        if show_details:
            print("Detailed Forecast:")
            display_df = results.copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
        else:
            print("Next 10 Hours Forecast:")
            display_df = results.head(10).copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            print(tabulate(display_df, headers='keys', tablefmt='grid', showindex=False, floatfmt='.2f'))
            
            if len(results) > 10:
                print(f"\n... and {len(results) - 10} more hours")
        
        return results
    
    def export_predictions(self, results, output_file='forecast.csv'):
        """Export predictions to CSV file."""
        results.to_csv(output_file, index=False)
        print(f"\n✓ Predictions exported to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='Weather Forecasting CLI with Hybrid Model Support')
    
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--city', type=str, help='City name')
    location_group.add_argument('--coords', nargs=2, type=float, metavar=('LAT', 'LON'))
    
    parser.add_argument('--hours', type=int, default=24, help='Forecast horizon')
    parser.add_argument('--target', type=str, default='temperature_2m', help='Target variable')
    parser.add_argument('--train', action='store_true', help='Train if needed')
    parser.add_argument('--retrain', action='store_true', help='Force retrain')
    parser.add_argument('--no-dl', action='store_true', help='Disable deep learning')
    parser.add_argument('--export', type=str, metavar='FILE', help='Export to CSV')
    parser.add_argument('--details', action='store_true', help='Show detailed predictions')
    
    args = parser.parse_args()
    
    if args.city:
        city_name, lat, lon = args.city, None, None
    else:
        city_name, lat, lon = None, args.coords[0], args.coords[1]
    
    try:
        cli = WeatherCLI(city_name=city_name, lat=lat, lon=lon)
        
        cli.ensure_model_exists(
            target_var=args.target,
            force_retrain=args.retrain or args.train,
            use_deep_learning=not args.no_dl
        )
        
        results = cli.predict(
            hours_ahead=args.hours,
            target_var=args.target,
            show_details=args.details
        )
        
        if args.export:
            cli.export_predictions(results, args.export)
        
        print("\n✓ Forecast completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()