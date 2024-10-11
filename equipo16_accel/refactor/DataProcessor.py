# src/data/data_processor.py

import pandas as pd
import numpy as np
import os

class DataProcessor:
    def __init__(self, data_dir=None):
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.join(current_dir, '..', 'data')
        else:
            self.data_dir = data_dir
        self.raw_data = None
        self.is_data_cleaned = False
        self.processed_data = None
    
    def load_data(self, filename='accelerometer.csv'):
        raw_data_path = os.path.join(self.data_dir, 'raw', filename)
        self.raw_data = pd.read_csv(raw_data_path)

    def clean_data(self):
        if self.raw_data is not None:
            self.raw_data = self.raw_data.dropna()
            self.is_data_cleaned = True
        else:
            raise ValueError("No data loaded. Please load data first.")

    def feature_engineering(self):
        if self.raw_data is not None or self.is_data_cleaned is False:
            df = self.raw_data.copy()
            # Create vibration magnitude feature
            df['vibration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
            # Map configurations
            df['configuración'] = df['wconfid'].map({1: 'Normal', 2: 'Perpendicular', 3: 'Opuesto'})
            # Adjust target labels
            df['wconfid'] -= 1
            self.processed_data = df
        else:
            raise ValueError("No data to process. Please load and clean data first.")

    def save_processed_data(self, filename='processed_data.csv'):
        if self.processed_data is not None:
            processed_data_path = os.path.join(self.data_dir, 'processed', filename)
            self.processed_data.to_csv(processed_data_path, index=False)
        else:
            raise ValueError("No processed data to save. Please run feature_engineering first.")
