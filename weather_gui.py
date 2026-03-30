import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
from datetime import datetime
import sys

# Import the training module
try:
    from train_download_weather import WeatherDataProcessor
    from weather_cli import WeatherCLI
except ImportError:
    print("❌ Error: train_download_weather.py and weather_cli.py must be in the same directory")
    sys.exit(1)


class WeatherForecastGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Weather Forecast System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Variables
        self.city_var = tk.StringVar(value="San Francisco")
        self.hours_var = tk.StringVar(value="24")
        self.target_var = tk.StringVar(value="temperature_2m")
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.DoubleVar(value=0)
        
        # Results storage
        self.current_results = None
        self.current_cli = None
        
        # Setup UI
        self.setup_ui()
        
        # Apply modern styling
        self.apply_styling()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Top section: Input controls
        self.create_input_section(main_frame)
        
        # Middle section: Results and visualization
        self.create_results_section(main_frame)
        
        # Bottom section: Status bar
        self.create_status_section(main_frame)
    
    def create_input_section(self, parent):
        """Create the input controls section."""
        input_frame = ttk.LabelFrame(parent, text="Forecast Settings", padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        # City input
        ttk.Label(input_frame, text="City Name:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        city_entry = ttk.Entry(input_frame, textvariable=self.city_var, width=30, font=('Arial', 11))
        city_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        # Forecast hours
        ttk.Label(input_frame, text="Forecast Hours:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        hours_frame = ttk.Frame(input_frame)
        hours_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        hours_entry = ttk.Entry(hours_frame, textvariable=self.hours_var, width=10, font=('Arial', 11))
        hours_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Quick select buttons
        ttk.Button(hours_frame, text="24h", command=lambda: self.hours_var.set("24"), width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(hours_frame, text="48h", command=lambda: self.hours_var.set("48"), width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(hours_frame, text="72h", command=lambda: self.hours_var.set("72"), width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(hours_frame, text="168h", command=lambda: self.hours_var.set("168"), width=6).pack(side=tk.LEFT, padx=2)
        
        # Target variable
        ttk.Label(input_frame, text="Target Variable:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        target_label = ttk.Label(input_frame, text="Temperature (°C)", font=('Arial', 10))
        target_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Action buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.forecast_btn = ttk.Button(button_frame, text="🌤 Generate Forecast", 
                                       command=self.generate_forecast, width=20)
        self.forecast_btn.pack(side=tk.LEFT, padx=5)
        
        self.retrain_btn = ttk.Button(button_frame, text="🔄 Retrain Model", 
                                      command=self.retrain_model, width=20)
        self.retrain_btn.pack(side=tk.LEFT, padx=5)
        
        self.export_btn = ttk.Button(button_frame, text="💾 Export CSV", 
                                     command=self.export_results, width=15, state='disabled')
        self.export_btn.pack(side=tk.LEFT, padx=5)
    
    def create_results_section(self, parent):
        """Create the results and visualization section."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(parent)
        self.notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Tab 1: Chart
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="📊 Chart")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Weather Forecast")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 2: Data table
        table_frame = ttk.Frame(self.notebook)
        self.notebook.add(table_frame, text="📋 Data Table")
        
        # Create treeview for data
        tree_container = ttk.Frame(table_frame)
        tree_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_container, orient="vertical")
        hsb = ttk.Scrollbar(tree_container, orient="horizontal")
        
        self.tree = ttk.Treeview(tree_container, yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)
        
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.pack(fill=tk.BOTH, expand=True)
        
        # Tab 3: Summary statistics
        summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(summary_frame, text="📈 Summary")
        
        self.summary_text = tk.Text(summary_frame, wrap=tk.WORD, font=('Courier', 11), 
                                   bg='#f5f5f5', padx=20, pady=20)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_status_section(self, parent):
        """Create the status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        status_frame.columnconfigure(0, weight=1)
        
        # Status label
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                           mode='determinate', length=200)
        self.progress_bar.grid(row=0, column=1, sticky=tk.E)
    
    def apply_styling(self):
        """Apply custom styling to the GUI."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TLabelframe', background='#f0f0f0')
        style.configure('TLabelframe.Label', font=('Arial', 11, 'bold'))
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('TLabel', font=('Arial', 10))
    
    def update_status(self, message, progress=None):
        """Update status bar message and progress."""
        self.status_var.set(message)
        if progress is not None:
            self.progress_var.set(progress)
        self.root.update_idletasks()
    
    def disable_buttons(self):
        """Disable action buttons during processing."""
        self.forecast_btn.config(state='disabled')
        self.retrain_btn.config(state='disabled')
    
    def enable_buttons(self):
        """Enable action buttons after processing."""
        self.forecast_btn.config(state='normal')
        self.retrain_btn.config(state='normal')
    
    def generate_forecast(self):
        """Generate weather forecast in a separate thread."""
        # Validate inputs
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
        
        try:
            hours = int(self.hours_var.get())
            if hours <= 0 or hours > 720:
                messagebox.showerror("Error", "Hours must be between 1 and 720")
                return
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of hours")
            return
        
        # Run in thread to keep GUI responsive
        thread = threading.Thread(target=self._generate_forecast_thread, 
                                 args=(city, hours, self.target_var.get()))
        thread.daemon = True
        thread.start()
    
    def _generate_forecast_thread(self, city, hours, target_var):
        """Worker thread for generating forecast."""
        try:
            self.disable_buttons()
            self.update_status(f"Initializing forecast for {city}...", 10)
            
            # Create CLI instance
            self.current_cli = WeatherCLI(city_name=city)
            
            self.update_status("Checking model availability...", 20)
            
            # Ensure model exists
            self.current_cli.ensure_model_exists(target_var=target_var)
            
            self.update_status("Generating predictions...", 50)
            
            # Generate predictions
            results = self.current_cli.predict(
                hours_ahead=hours,
                target_var=target_var,
                show_details=False
            )
            
            self.current_results = results
            
            self.update_status("Updating visualizations...", 80)
            
            # Update UI
            self.root.after(0, self.update_results_display, results, city, target_var)
            
            self.update_status(f"✓ Forecast completed for {city}", 100)
            
            # Enable export button
            self.root.after(0, lambda: self.export_btn.config(state='normal'))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Forecast failed:\n{error_msg}"))
            self.update_status("❌ Forecast failed", 0)
        finally:
            self.root.after(0, self.enable_buttons)
    
    def update_results_display(self, results, city, target_var):
        """Update the results display with forecast data."""
        # Update chart
        self.ax.clear()
        self.ax.plot(results['Timestamp'], results['Predicted'], 
                    marker='o', linewidth=2, markersize=4, color='#2E86AB', label='Predicted')
        self.ax.set_title(f"{city} - {target_var.replace('_', ' ').title()} Forecast", 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Date/Time", fontsize=11)
        self.ax.set_ylabel(target_var.replace('_', ' ').title(), fontsize=11)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.fig.autofmt_xdate()
        self.canvas.draw()
        
        # Update data table
        self.tree.delete(*self.tree.get_children())
        
        # Configure columns
        self.tree['columns'] = ('Timestamp', 'Predicted')
        self.tree['show'] = 'headings'
        
        self.tree.heading('Timestamp', text='Date/Time')
        self.tree.heading('Predicted', text=f'Predicted {target_var}')
        
        self.tree.column('Timestamp', width=200)
        self.tree.column('Predicted', width=150)
        
        # Add data
        for _, row in results.iterrows():
            self.tree.insert('', tk.END, values=(
                row['Timestamp'].strftime('%Y-%m-%d %H:%M'),
                f"{row['Predicted']:.2f}"
            ))
        
        # Update summary
        self.summary_text.delete('1.0', tk.END)
        
        summary = f"""
{'='*60}
FORECAST SUMMARY
{'='*60}

City: {city}
Target Variable: {target_var.replace('_', ' ').title()}
Forecast Duration: {len(results)} hours
Start Time: {results['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}
End Time: {results['Timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}

{'='*60}
STATISTICS
{'='*60}

Mean Value:       {results['Predicted'].mean():.2f}
Minimum Value:    {results['Predicted'].min():.2f}
Maximum Value:    {results['Predicted'].max():.2f}
Standard Dev:     {results['Predicted'].std():.2f}
Median Value:     {results['Predicted'].median():.2f}

{'='*60}
HOURLY BREAKDOWN (First 24 Hours)
{'='*60}

"""
        self.summary_text.insert('1.0', summary)
        
        # Add hourly data
        for i, row in results.head(24).iterrows():
            hour_text = f"{row['Timestamp'].strftime('%Y-%m-%d %H:%M')}    {row['Predicted']:.2f}\n"
            self.summary_text.insert(tk.END, hour_text)
        
        if len(results) > 24:
            self.summary_text.insert(tk.END, f"\n... and {len(results) - 24} more hours\n")
    
    def retrain_model(self):
        """Retrain the model with fresh data."""
        city = self.city_var.get().strip()
        if not city:
            messagebox.showerror("Error", "Please enter a city name")
            return
        
        result = messagebox.askyesno("Confirm Retrain", 
                                     f"This will download fresh data and retrain the model for {city}.\n\n"
                                     "This may take several minutes. Continue?")
        if not result:
            return
        
        thread = threading.Thread(target=self._retrain_model_thread, 
                                 args=(city, self.target_var.get()))
        thread.daemon = True
        thread.start()
    
    def _retrain_model_thread(self, city, target_var):
        """Worker thread for retraining model."""
        try:
            self.disable_buttons()
            self.update_status("Downloading fresh data...", 20)
            
            processor = WeatherDataProcessor(city_name=city)
            
            self.update_status("Training model...", 40)
            
            processor.run_full_pipeline(target_var=target_var, force_refresh=True)
            
            self.update_status("✓ Model retrained successfully", 100)
            
            self.root.after(0, lambda: messagebox.showinfo("Success", 
                                                           f"Model for {city} has been retrained with fresh data!"))
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Error", f"Retraining failed:\n{error_msg}"))
            self.update_status("❌ Retraining failed", 0)
        finally:
            self.root.after(0, self.enable_buttons)
    
    def export_results(self):
        """Export results to CSV file."""
        if self.current_results is None:
            messagebox.showwarning("No Data", "Please generate a forecast first")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{self.city_var.get().lower().replace(' ', '_')}_forecast_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        )
        
        if filename:
            try:
                self.current_results.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Forecast exported to:\n{filename}")
                self.update_status(f"✓ Exported to {filename}", 100)
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    app = WeatherForecastGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()