import pandas as pd
import numpy as np
import pickle
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from tabulate import tabulate
import sys

# Import the training module
try:
    from train_download_weather import WeatherDataProcessor
except ImportError:
    print("❌ Error: train_download_weather.py must be in the same directory")
    sys.exit(1)


class WeatherCLI:
    """Command-line interface for weather forecasting."""
    
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
    
    def ensure_model_exists(self, target_var='temperature_2m', force_retrain=False):
        """Ensure that a model exists for the city, train if necessary."""
        model_path = self.processor._get_model_path(target_var)
        cache_path = self.processor._get_cache_path()
        
        # Check if model exists
        model_exists = model_path.exists()
        cache_exists = cache_path.exists()
        
        # Check if cache is fresh
        cache_is_fresh = self.processor._is_cache_valid()
        
        # Decide whether to train
        should_train = False
        
        if not model_exists:
            print(f"⚠ No model found for {self.city_name or 'location'}")
            should_train = True
        elif force_retrain:
            print(f"⚠ Force retrain requested")
            should_train = True
        elif not cache_is_fresh and cache_exists:
            print(f"⚠ Cache is outdated, will retrain with fresh data")
            should_train = True
        
        if should_train:
            print(f"\n⏳ Training model for {self.city_name or 'location'}...")
            self.processor.run_full_pipeline(
                target_var=target_var,
                force_refresh=not cache_is_fresh
            )
            print("✓ Model training completed\n")
        else:
            print(f"✓ Using existing model for {self.city_name or 'location'}")
    
    def load_model(self, target_var='temperature_2m'):
        """Load the trained model and metadata."""
        model_path = self.processor._get_model_path(target_var)
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Run with --train flag to create a model."
            )
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return model_data
    
    def load_and_prepare_data(self, target_var='temperature_2m', lag_hours=[1, 2, 3, 6, 12, 24, 48, 72, 168]):
        """Load cached data and prepare features."""
        cache_path = self.processor._get_cache_path()
        
        if not cache_path.exists():
            print(f"⚠ No cached data found, downloading...")
            df = self.processor.download_weather_data()
        else:
            df = pd.read_csv(cache_path, parse_dates=['date'])
        
        # Preprocess data
        df_processed = self.processor.preprocess_data(
            df, 
            target_var=target_var, 
            lag_hours=lag_hours
        )
        
        return df_processed
    
    def predict(self, hours_ahead=24, target_var='temperature_2m', show_details=False):
        """Generate weather predictions for future hours using iterative approach with pattern preservation."""
        print(f"\n{'='*60}")
        print(f"Weather Forecast")
        print(f"City: {self.city_name or f'({self.lat}, {self.lon})'}")
        print(f"Target: {target_var}")
        print(f"Horizon: {hours_ahead} hours into the future")
        print(f"{'='*60}\n")
        
        # Load model
        print("⏳ Loading model...")
        model_data = self.load_model(target_var)
        model = model_data['model']
        feature_cols = model_data['feature_cols']
        
        print(f"✓ Model loaded (R² score: {model_data['avg_score']:.4f})")
        print(f"  Trained at: {model_data['trained_at']}")
        
        # Load and prepare data
        print("\n⏳ Loading historical data...")
        df = self.load_and_prepare_data(target_var=target_var)
        
        if len(df) == 0:
            raise ValueError("No data available for prediction")
        
        # Get the most recent complete data point
        last_timestamp = df.index[-1]
        print(f"  Last known data: {last_timestamp}")
        
        # Start forecasting from the next hour
        forecast_start = last_timestamp + pd.Timedelta(hours=1)
        print(f"  Forecasting from: {forecast_start}")
        
        # Calculate daily pattern from recent history (last 30 days)
        recent_history = df.tail(min(720, len(df)))  # Last 30 days
        hourly_patterns = {}
        
        # Calculate average temperature for each hour of day
        for hour in range(24):
            hour_data = recent_history[recent_history.index.hour == hour][target_var]
            if len(hour_data) > 0:
                hourly_patterns[hour] = {
                    'mean': hour_data.mean(),
                    'std': hour_data.std()
                }
        
        # Calculate day-to-day trend (warming or cooling)
        if len(recent_history) >= 168:  # At least 1 week
            weekly_temps = recent_history[target_var].tail(168)
            daily_trend = (weekly_temps.iloc[-24:].mean() - weekly_temps.iloc[:24].mean()) / 7
        else:
            daily_trend = 0
        
        # Rolling prediction: predict one step at a time
        print(f"\n⏳ Generating {hours_ahead}-hour forecast...")
        predictions = []
        forecast_times = []
        
        # Get lag column names
        lag_cols = [col for col in feature_cols if f'{target_var}_lag_' in col]
        
        # Prepare a working dataframe
        working_df = recent_history.tail(100).copy()
        
        for i in range(hours_ahead):
            # Get the most recent row
            current_row = working_df.iloc[-1:].copy()
            
            # Calculate forecast timestamp
            forecast_time = forecast_start + pd.Timedelta(hours=i)
            
            # Update lag features based on recent history
            for lag_col in lag_cols:
                lag_hours = int(lag_col.split('_lag_')[1].replace('h', ''))
                
                if i >= lag_hours:
                    # Use our predictions
                    current_row[lag_col] = predictions[i - lag_hours]
                elif len(working_df) >= lag_hours:
                    # Use historical data
                    current_row[lag_col] = working_df.iloc[-(lag_hours + 1)][target_var]
            
            # Update time-based features
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
            
            # Update rolling mean features
            if f'{target_var}_rolling_mean_6h' in current_row.columns:
                if i >= 6:
                    current_row[f'{target_var}_rolling_mean_6h'] = np.mean(predictions[-6:])
                else:
                    recent_vals = list(working_df[target_var].tail(6 - i)) + predictions[:i]
                    current_row[f'{target_var}_rolling_mean_6h'] = np.mean(recent_vals)
            
            if f'{target_var}_rolling_mean_24h' in current_row.columns:
                if i >= 24:
                    current_row[f'{target_var}_rolling_mean_24h'] = np.mean(predictions[-24:])
                else:
                    recent_vals = list(working_df[target_var].tail(24 - i)) + predictions[:i]
                    current_row[f'{target_var}_rolling_mean_24h'] = np.mean(recent_vals)
            
            # Update difference features
            if f'{target_var}_diff_1h' in current_row.columns:
                if i > 0:
                    current_row[f'{target_var}_diff_1h'] = predictions[-1] - (predictions[-2] if i > 1 else working_df.iloc[-1][target_var])
                else:
                    current_row[f'{target_var}_diff_1h'] = working_df.iloc[-1][f'{target_var}_diff_1h']
            
            if f'{target_var}_diff_24h' in current_row.columns:
                if i >= 24:
                    current_row[f'{target_var}_diff_24h'] = predictions[-1] - predictions[-24]
                else:
                    current_row[f'{target_var}_diff_24h'] = working_df.iloc[-1][f'{target_var}_diff_24h']
            
            # For other weather features, use persistence with daily pattern
            if i > 0:
                # Get reference from 24 hours ago for daily cycling
                hours_back = min(24, len(working_df) - 1)
                reference_row = working_df.iloc[-hours_back]
                
                # Blend: 50% recent + 50% daily pattern
                for col in current_row.columns:
                    if (col not in lag_cols and 
                        col != target_var and 
                        'rolling_mean' not in col and
                        'diff' not in col and
                        col not in ['hour', 'day_of_week', 'month', 'day_of_year', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'] and
                        col in reference_row.index):
                        if pd.notna(current_row[col].iloc[0]) and pd.notna(reference_row[col]):
                            recent_val = current_row[col].iloc[0]
                            pattern_val = reference_row[col]
                            # Gradual blend toward daily pattern
                            blend_factor = min(0.7, 0.3 + (i / hours_ahead) * 0.4)
                            blended = (1 - blend_factor) * recent_val + blend_factor * pattern_val
                            current_row[col] = blended
            
            # Ensure all required features are present
            missing_cols = set(feature_cols) - set(current_row.columns)
            if missing_cols:
                for col in missing_cols:
                    if col in working_df.columns:
                        current_row[col] = working_df[col].mean()
                    else:
                        current_row[col] = 0
            
            # Prepare features for prediction
            X = current_row[feature_cols]
            
            # Make prediction
            pred = model.predict(X)[0]
            
            # Apply pattern correction for long-term forecasts
            if i >= 48:  # After 2 days, start applying pattern correction
                hour_of_day = forecast_time.hour
                if hour_of_day in hourly_patterns:
                    pattern_mean = hourly_patterns[hour_of_day]['mean']
                    # Gradually blend prediction with historical pattern
                    correction_strength = min(0.5, (i - 48) / hours_ahead)
                    pred = (1 - correction_strength) * pred + correction_strength * pattern_mean
                
                # Apply long-term trend
                days_ahead = i / 24
                pred += daily_trend * days_ahead
            
            predictions.append(pred)
            forecast_times.append(forecast_time)
            
            # Add this prediction to working dataframe
            new_row = current_row.copy()
            new_row.index = [forecast_time]
            new_row[target_var] = pred
            working_df = pd.concat([working_df, new_row])
            
            # Keep only recent history
            if len(working_df) > 200:
                working_df = working_df.tail(200)
        
        # Create results dataframe
        results = pd.DataFrame({
            'Timestamp': forecast_times,
            'Predicted': predictions,
        })
        
        # Calculate summary statistics
        print(f"\n{'='*60}")
        print("Forecast Summary")
        print(f"{'='*60}")
        print(f"Mean predicted {target_var}: {np.mean(predictions):.2f}")
        print(f"Min predicted {target_var}: {np.min(predictions):.2f}")
        print(f"Max predicted {target_var}: {np.max(predictions):.2f}")
        print(f"Std deviation: {np.std(predictions):.2f}")
        print(f"{'='*60}\n")
        
        # Display detailed predictions
        if show_details:
            print("Detailed Forecast:")
            display_df = results.copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            print(tabulate(
                display_df,
                headers='keys',
                tablefmt='grid',
                showindex=False,
                floatfmt='.2f'
            ))
        else:
            # Show first 10 predictions
            print("Next 10 Hours Forecast:")
            display_df = results.head(10).copy()
            display_df['Timestamp'] = display_df['Timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            print(tabulate(
                display_df,
                headers='keys',
                tablefmt='grid',
                showindex=False,
                floatfmt='.2f'
            ))
            
            if len(results) > 10:
                print(f"\n... and {len(results) - 10} more hours")
                print(f"(Use --details flag to see all {hours_ahead} hours)")
        
        return results
    
    def export_predictions(self, results, output_file='forecast.csv'):
        """Export predictions to CSV file."""
        results.to_csv(output_file, index=False)
        print(f"\n✓ Predictions exported to {output_file}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Weather Forecasting CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get 24-hour forecast for a city
  python weather_cli.py --city "San Francisco" --hours 24
  
  # Train a new model and predict
  python weather_cli.py --city "New York" --train --hours 48
  
  # Force retrain with fresh data
  python weather_cli.py --city "London" --retrain --hours 24
  
  # Use coordinates instead of city name
  python weather_cli.py --lat 37.7749 --lon -122.4194 --hours 12
  
  # Export predictions to CSV
  python weather_cli.py --city "Tokyo" --hours 72 --export forecast_tokyo.csv
  
  # Show detailed predictions
  python weather_cli.py --city "Paris" --hours 24 --details
        """
    )
    
    # Location arguments
    location_group = parser.add_mutually_exclusive_group(required=True)
    location_group.add_argument('--city', type=str, help='City name')
    location_group.add_argument('--coords', nargs=2, type=float, metavar=('LAT', 'LON'),
                               help='Coordinates (latitude longitude)')
    
    # Prediction arguments
    parser.add_argument('--hours', type=int, default=24,
                       help='Forecast horizon in hours (default: 24)')
    parser.add_argument('--target', type=str, default='temperature_2m',
                       help='Target variable to predict (default: temperature_2m)')
    
    # Model management arguments
    parser.add_argument('--train', action='store_true',
                       help='Train a new model if one doesn\'t exist')
    parser.add_argument('--retrain', action='store_true',
                       help='Force retrain model with latest data')
    
    # Output arguments
    parser.add_argument('--export', type=str, metavar='FILE',
                       help='Export predictions to CSV file')
    parser.add_argument('--details', action='store_true',
                       help='Show detailed predictions for all hours')
    
    args = parser.parse_args()
    
    # Parse location
    if args.city:
        city_name = args.city
        lat, lon = None, None
    else:
        city_name = None
        lat, lon = args.coords
    
    try:
        # Initialize CLI
        cli = WeatherCLI(city_name=city_name, lat=lat, lon=lon)
        
        # Ensure model exists
        cli.ensure_model_exists(
            target_var=args.target,
            force_retrain=args.retrain or args.train
        )
        
        # Generate predictions
        results = cli.predict(
            hours_ahead=args.hours,
            target_var=args.target,
            show_details=args.details
        )
        
        # Export if requested
        if args.export:
            cli.export_predictions(results, args.export)
        
        print("\n✓ Forecast completed successfully!\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()