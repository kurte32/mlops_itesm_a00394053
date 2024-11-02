# tests/test_model_trainer_pytest.py

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from refactor.ModelTrainer import ModelTrainer


@pytest.fixture
def synthetic_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]

@pytest.fixture
def model_trainer(synthetic_data):
    x_train, x_test, y_train, y_test = synthetic_data
    model = LogisticRegression()
    params = {'C': 1.0, 'solver': 'liblinear'}
    trainer = ModelTrainer(
        model=model,
        params=params,
        model_name='LogisticRegression',
        mlflow_experiment='Test_Experiment'
    )
    return trainer, x_train, x_test, y_train, y_test

@patch('src.models.ModelTrainer.mlflow.set_tracking_uri')
@patch('src.models.ModelTrainer.mlflow.set_experiment')
def test_initialization(mock_set_experiment, mock_set_tracking_uri, model_trainer, synthetic_data):
    trainer, _, _, _, _ = model_trainer
    mock_set_tracking_uri.assert_called_once_with(trainer.mlflow_tracking_uri)
    mock_set_experiment.assert_called_once_with(trainer.mlflow_experiment)
    assert trainer.model_name == 'LogisticRegression'
    assert trainer.params == {'C': 1.0, 'solver': 'liblinear'}

@patch.object(LogisticRegression, 'fit')
@patch.object(LogisticRegression, 'set_params')
def test_train_success(mock_set_params, mock_fit, model_trainer, synthetic_data):
    trainer, x_train, _, _, _ = model_trainer
    trainer._train(x_train, synthetic_data[2])
    mock_set_params.assert_called_once_with(**trainer.params)
    mock_fit.assert_called_once_with(x_train, synthetic_data[2])

@patch('src.models.ModelTrainer.mlflow.log_metric')
def test_log_metrics_success(mock_log_metric, model_trainer, synthetic_data):
    trainer, _, _, _, _ = model_trainer
    metrics = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.77}
    trainer._log_metrics(metrics)
    for key, value in metrics.items():
        mock_log_metric.assert_any_call(key, value)
    assert mock_log_metric.call_count == len(metrics)
