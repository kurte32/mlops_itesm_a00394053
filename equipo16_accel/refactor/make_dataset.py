# src/data/make_dataset.py

import pandas as pd
import numpy as np
import os

def load_raw_data(data_dir):
    raw_data_path = os.path.join(data_dir, 'raw', 'accelerometer.csv')
    df = pd.read_csv(raw_data_path)
    return df

def preprocess_data(df):
    # Data cleaning
    df = df.dropna()
    
    # Feature engineering
    df['vibration_magnitude'] = np.sqrt(df['x']**2 + df['y']**2 + df['z']**2)
    df['configuración'] = df['wconfid'].map({1: 'Normal', 2: 'Perpendicular', 3: 'Opuesto'})
    df['wconfid'] -= 1  # Adjust target to start from 0
    return df

def save_processed_data(df, data_dir):
    processed_data_path = os.path.join(data_dir, 'processed', 'processed_data.csv')
    df.to_csv(processed_data_path, index=False)

def main():
    data_dir = '/home/kurtbadelt/MNA/MLOPS/MLOps/notebooks/Actividades/Fase1/equipo16_accel/data'
    df = load_raw_data(data_dir)
    df = preprocess_data(df)
    save_processed_data(df, data_dir)

if __name__ == '__main__':
    main()
