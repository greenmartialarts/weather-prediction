from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os

# Import our modules
from train_download_weather import WeatherDataProcessor
from weather_cli import WeatherCLI

app = Flask(__name__)
CORS(app)

# Store active sessions
sessions = {}


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/forecast', methods=['POST'])
def generate_forecast():
    """Generate a weather forecast."""
    try:
        data = request.json
        city = data.get('city', '').strip()
        hours = int(data.get('hours', 24))
        
        if not city:
            return jsonify({'error': 'City name is required'}), 400
        
        if hours <= 0 or hours > 720:
            return jsonify({'error': 'Hours must be between 1 and 720'}), 400
        
        # Create CLI instance
        cli = WeatherCLI(city_name=city)
        
        # Ensure model exists
        cli.ensure_model_exists(target_var='temperature_2m')
        
        # Generate predictions
        results = cli.predict(
            hours_ahead=hours,
            target_var='temperature_2m',
            show_details=False
        )
        
        # Prepare response data
        forecast_data = {
            'city': city,
            'hours': hours,
            'timestamps': results['Timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            'temperatures': results['Predicted'].round(2).tolist(),
            'stats': {
                'mean': float(results['Predicted'].mean()),
                'min': float(results['Predicted'].min()),
                'max': float(results['Predicted'].max()),
                'std': float(results['Predicted'].std()),
                'median': float(results['Predicted'].median()),
                'range': float(results['Predicted'].max() - results['Predicted'].min())
            }
        }
        
        # Generate chart
        chart_data = generate_chart(results, city)
        forecast_data['chart'] = chart_data
        
        return jsonify(forecast_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with fresh data."""
    try:
        data = request.json
        city = data.get('city', '').strip()
        
        if not city:
            return jsonify({'error': 'City name is required'}), 400
        
        processor = WeatherDataProcessor(city_name=city)
        processor.run_full_pipeline(target_var='temperature_2m', force_refresh=True)
        
        return jsonify({'message': f'Model for {city} retrained successfully'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def export_csv():
    """Export forecast data as CSV."""
    try:
        data = request.json
        timestamps = data.get('timestamps', [])
        temperatures = data.get('temperatures', [])
        
        df = pd.DataFrame({
            'Timestamp': timestamps,
            'Temperature_C': temperatures
        })
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        return jsonify({
            'csv': output.getvalue()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def generate_chart(results, city):
    """Generate a chart image and return as base64."""
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    
    # Plot
    ax.plot(results['Timestamp'], results['Predicted'], 
            linewidth=2.5, color='#3B82F6', marker='o', 
            markersize=4, markerfacecolor='white', 
            markeredgewidth=2, markeredgecolor='#3B82F6')
    
    # Styling
    ax.set_title(f'{city} - Temperature Forecast', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date/Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f'data:image/png;base64,{image_base64}'


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=8000)