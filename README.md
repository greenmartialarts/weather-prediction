# Weather Forecasting System

A fully automated, city-specific weather forecasting system that downloads historical data, trains XGBoost models, and generates predictions with minimal user intervention.

## Features

- 🌍 **Multi-City Support**: Each city has its own cache and model
- 📊 **Comprehensive Data**: 25+ weather variables including temperature, humidity, wind, precipitation, and more
- 🤖 **Automatic Training**: Models are automatically trained when needed
- 💾 **Smart Caching**: Efficient data storage with automatic updates
- 📈 **Time Series Forecasting**: Uses lag features and XGBoost for accurate predictions
- 🔄 **Auto-Updates**: Refreshes data and retrains models when cache is outdated

## Installation

### Prerequisites

```bash
python >= 3.7
```

### Install Dependencies

```bash
pip install pandas numpy requests scikit-learn xgboost tabulate
```

### File Structure

```
weather_forecasting/
├── train_download_weather.py   # Core training pipeline
├── weather_cli.py               # Command-line interface
├── weather_cache/               # Auto-created: stores city data
├── weather_models/              # Auto-created: stores trained models
└── README.md
```

## Quick Start

### 1. Basic Forecast (24 hours)

```bash
python weather_cli.py --city "San Francisco" --hours 24
```

On first run, this will:
- Download 1 year of historical weather data
- Train a city-specific model
- Generate 24-hour temperature forecast

### 2. Longer Forecast

```bash
python weather_cli.py --city "New York" --hours 72
```

### 3. Using Coordinates

```bash
python weather_cli.py --coords 37.7749 -122.4194 --hours 24
```

### 4. Detailed Output

```bash
python weather_cli.py --city "London" --hours 48 --details
```

## Advanced Usage

### Training Module (`train_download_weather.py`)

#### Command Line

```bash
# Train model for a specific city
python train_download_weather.py --city "Tokyo" --days 365

# Use coordinates
python train_download_weather.py --lat 35.6762 --lon 139.6503 --days 365

# Force refresh cache
python train_download_weather.py --city "Paris" --force

# Train for different target variable
python train_download_weather.py --city "Berlin" --target "relative_humidity_2m"
```

#### Python API

```python
from train_download_weather import WeatherDataProcessor

# Initialize processor
processor = WeatherDataProcessor(city_name="Seattle", cache_days=365)

# Run full pipeline
results = processor.run_full_pipeline(
    target_var='temperature_2m',
    lag_hours=[1, 2, 3, 24],
    force_refresh=False
)

# Access components
model = results['model']
features = results['features']
score = results['score']
data = results['data']
```

### CLI Module (`weather_cli.py`)

#### Basic Commands

```bash
# Simple forecast
python weather_cli.py --city "Boston" --hours 24

# Force retrain with latest data
python weather_cli.py --city "Chicago" --retrain --hours 48

# Export to CSV
python weather_cli.py --city "Miami" --hours 72 --export miami_forecast.csv

# Show all predictions
python weather_cli.py --city "Seattle" --hours 168 --details
```

#### Python API

```python
from weather_cli import WeatherCLI

# Initialize CLI
cli = WeatherCLI(city_name="Los Angeles")

# Ensure model exists (auto-trains if needed)
cli.ensure_model_exists(target_var='temperature_2m')

# Generate predictions
results = cli.predict(hours_ahead=48, show_details=True)

# Export results
cli.export_predictions(results, 'la_forecast.csv')
```

## Configuration Options

### Weather Variables Available

The system downloads 25+ weather variables:

**Temperature & Humidity**
- `temperature_2m`: Air temperature at 2m
- `apparent_temperature`: Feels-like temperature
- `dew_point_2m`: Dew point temperature
- `relative_humidity_2m`: Relative humidity
- `vapour_pressure_deficit`: VPD

**Precipitation**
- `precipitation`: Total precipitation
- `rain`: Rainfall
- `snowfall`: Snowfall amount
- `snow_depth`: Snow depth

**Wind**
- `wind_speed_10m`: Wind speed at 10m
- `wind_direction_10m`: Wind direction
- `wind_gusts_10m`: Wind gusts

**Clouds & Visibility**
- `cloud_cover`: Total cloud cover
- `cloud_cover_low/mid/high`: Cloud cover by level
- `visibility`: Visibility distance

**Pressure & Other**
- `pressure_msl`: Mean sea level pressure
- `surface_pressure`: Surface pressure
- `weather_code`: WMO weather code
- `et0_fao_evapotranspiration`: Evapotranspiration

**Soil Conditions**
- `soil_temperature_0_to_7cm`: Shallow soil temp
- `soil_temperature_7_to_28cm`: Deep soil temp
- `soil_moisture_0_to_7cm`: Shallow soil moisture
- `soil_moisture_7_to_28cm`: Deep soil moisture

### Model Hyperparameters

You can customize XGBoost parameters in the training pipeline:

```python
custom_params = {
    'objective': 'reg:squarederror',
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 200,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 42
}

processor = WeatherDataProcessor(city_name="Denver")
processor.run_full_pipeline(xgb_params=custom_params)
```

### Lag Features

Control temporal features for better predictions:

```python
# Use 1, 2, 3, and 24 hour lags (default)
lag_hours = [1, 2, 3, 24]

# Or customize
lag_hours = [1, 3, 6, 12, 24, 48]  # More historical context
```

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     weather_cli.py                          │
│              (User-Facing Interface)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ Uses
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              train_download_weather.py                      │
│           (Core Training Pipeline)                          │
├─────────────────────────────────────────────────────────────┤
│  1. Data Download    → Open-Meteo Archive API              │
│  2. Preprocessing    → Lag features, missing value handling│
│  3. Model Training   → XGBoost with time series CV         │
│  4. Caching          → City-specific CSV storage           │
│  5. Model Storage    → City-specific .pkl files            │
└─────────────────────────────────────────────────────────────┘
```

### Workflow

1. **First Run for a City**:
   - CLI checks for existing model → Not found
   - Downloads 365 days of historical data from Open-Meteo
   - Saves to `weather_cache/{city}_weather_cache.csv`
   - Creates lag features (1h, 2h, 3h, 24h back)
   - Trains XGBoost model with time series cross-validation
   - Saves model to `weather_models/{city}_temperature_model.pkl`
   - Generates predictions

2. **Subsequent Runs**:
   - CLI checks cache age → If < 1 day old, use cached data
   - Loads existing model
   - Generates predictions immediately

3. **Auto-Update**:
   - If cache > 1 day old → Downloads fresh data
   - Retrains model with updated data
   - Generates predictions with new model

### Time Series Cross-Validation

The system uses `TimeSeriesSplit` to respect temporal ordering:

```
Fold 1: [Train-----] [Val]
Fold 2: [Train----------] [Val]
Fold 3: [Train---------------] [Val]
Fold 4: [Train--------------------] [Val]
Fold 5: [Train-------------------------] [Val]
```

This prevents data leakage and provides realistic performance estimates.

## File Outputs

### Cache Files

Located in `weather_cache/`:
- Format: `{city_name}_weather_cache.csv`
- Example: `san_francisco_weather_cache.csv`
- Contains: ~8,760 rows (1 year of hourly data)
- Size: ~2-5 MB per city

### Model Files

Located in `weather_models/`:
- Format: `{city_name}_{target_var}_model.pkl`
- Example: `new_york_temperature_2m_model.pkl`
- Contains: Trained XGBoost model + metadata
- Size: ~1-3 MB per model

## Troubleshooting

### Error: "Could not find coordinates for city"

**Solution**: City name might be ambiguous. Try using coordinates:
```bash
python weather_cli.py --coords 40.7128 -74.0060 --hours 24
```

### Error: "No data remaining after preprocessing"

**Cause**: Dataset too small or too many missing values.

**Solution**: Reduce lag hours or increase cache days:
```bash
python train_download_weather.py --city "YourCity" --days 730
```

### Error: "Model not found"

**Solution**: Run with `--train` flag to create model:
```bash
python weather_cli.py --city "YourCity" --train --hours 24
```

### Poor Prediction Accuracy

**Solutions**:
1. Retrain with more data: `--days 730` (2 years)
2. Tune hyperparameters in the code
3. Add more lag features
4. Try different target variables

## Performance Tips

1. **Parallel Multi-City**: Each city is independent - you can run multiple instances
2. **Cache Management**: Delete old caches periodically to save disk space
3. **Model Retraining**: Retrain weekly for best accuracy with `--retrain` flag
4. **Long Forecasts**: Accuracy decreases for horizons > 72 hours

## Limitations

- Historical data limited to Open-Meteo availability (typically 1979-present)
- Forecast accuracy degrades beyond 48-72 hours
- Requires internet connection for initial data download
- Cache refresh requires API access
- Models are location-specific and don't transfer between cities

## Data Source

Weather data provided by [Open-Meteo](https://open-meteo.com/):
- Archive API for historical data
- Geocoding API for city coordinates
- Free tier: Unlimited non-commercial use
- No API key required

## License

This project is provided as-is for educational and personal use.

## Contributing

Suggestions for improvement:
- Add support for ensemble models
- Implement uncertainty quantification
- Add visualization tools
- Support for additional data sources
- Real-time forecast updates

## Example Outputs

### Basic Forecast
```
============================================================
Weather Forecast
City: San Francisco
Target: temperature_2m
Horizon: 24 hours
============================================================

✓ Model loaded (R² score: 0.9234)
  Trained at: 2025-01-15T10:30:00

============================================================
Forecast Summary
============================================================
Mean predicted temperature_2m: 15.43
Min predicted temperature_2m: 12.87
Max predicted temperature_2m: 18.92
Std deviation: 1.84
============================================================

Next 10 Hours Forecast:
+---------------------+-----------+
| Timestamp           | Predicted |
+=====================+===========+
| 2025-01-15 11:00    | 15.23     |
| 2025-01-15 12:00    | 15.67     |
| 2025-01-15 13:00    | 16.12     |
| 2025-01-15 14:00    | 16.89     |
| 2025-01-15 15:00    | 17.45     |
| 2025-01-15 16:00    | 18.02     |
| 2025-01-15 17:00    | 17.86     |
| 2025-01-15 18:00    | 16.92     |
| 2025-01-15 19:00    | 15.34     |
| 2025-01-15 20:00    | 14.18     |
+---------------------+-----------+

✓ Forecast completed successfully!
```

## Support

For issues or questions:
1. Check this README for common solutions
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify internet connectivity for downloads

---

**Happy Forecasting! 🌤️**