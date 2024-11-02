# tests/test_data_processor.py

import pytest
from unittest.mock import patch
import pandas as pd
import numpy as np
import os
import tempfile
from refactor.DataProcessor import DataProcessor

@pytest.fixture
def data_processor():
    with tempfile.TemporaryDirectory() as test_dir:
        data_dir = test_dir
        sample_data = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'z': [7, 8, 9],
            'wconfid': [1, 2, 3]
        })
        raw_data_path = os.path.join(data_dir, 'raw', 'accelerometer.csv')
        os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
        sample_data.to_csv(raw_data_path, index=False)
        yield DataProcessor(data_dir=data_dir)

def test_load_data_success(data_processor):
    data_processor.load_data()
    expected_data = pd.read_csv(os.path.join(data_processor.data_dir, 'raw', 'accelerometer.csv'))
    pd.testing.assert_frame_equal(data_processor.raw_data, expected_data)

@patch('src.data.data_processor.pd.read_csv')
def test_load_data_file_not_found(mock_read_csv, data_processor):
    mock_read_csv.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        data_processor.load_data('non_existent.csv')

def test_clean_data_success(data_processor):
    data_processor.load_data()
    # Introduce NaN values
    data_processor.raw_data.loc[1, 'x'] = np.nan
    data_processor.clean_data()
    expected_data = data_processor.raw_data.dropna()
    pd.testing.assert_frame_equal(data_processor.raw_data, expected_data)
    assert data_processor.is_data_cleaned

def test_clean_data_no_data_loaded():
    processor = DataProcessor()
    with pytest.raises(ValueError, match="No data loaded"):
        processor.clean_data()

def test_feature_engineering_success(data_processor):
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.feature_engineering()

    expected_magnitude = np.sqrt(
        data_processor.raw_data['x']**2 +
        data_processor.raw_data['y']**2 +
        data_processor.raw_data['z']**2
    )
    pd.testing.assert_series_equal(
        data_processor.processed_data['vibration_magnitude'],
        expected_magnitude,
        check_names=False
    )

    expected_config = data_processor.raw_data['wconfid'].map({1: 'Normal', 2: 'Perpendicular', 3: 'Opuesto'})
    pd.testing.assert_series_equal(
        data_processor.processed_data['configuración'],
        expected_config,
        check_names=False
    )

    expected_wconfid = data_processor.raw_data['wconfid'] - 1
    pd.testing.assert_series_equal(
        data_processor.processed_data['wconfid'],
        expected_wconfid,
        check_names=False
    )

def test_feature_engineering_no_data_cleaned(data_processor):
    data_processor.load_data()
    with pytest.raises(ValueError, match="No data to process"):
        data_processor.feature_engineering()

def test_save_processed_data_success(data_processor):
    data_processor.load_data()
    data_processor.clean_data()
    data_processor.feature_engineering()

    data_processor.save_processed_data()
    saved_path = os.path.join(data_processor.data_dir, 'processed', 'processed_data.csv')
    saved_data = pd.read_csv(saved_path)
    pd.testing.assert_frame_equal(saved_data, data_processor.processed_data)

def test_save_processed_data_no_data():
    processor = DataProcessor()
    with pytest.raises(ValueError, match="No processed data to save"):
        processor.save_processed_data()
