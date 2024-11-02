# tests/test_model_trainer_pytest.py

import pytest
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from refactor.ModelTrainer import ModelTrainer

@pytest.fixture
def synthetic_data():
    """
    Fixture to create synthetic classification data.
    Returns:
        Tuple containing training and testing data.
    """
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X[:80], X[80:], y[:80], y[80:]

@pytest.fixture
def mock_mlflow():
    """
    Fixture to mock MLflow functions.
    Yields:
        Tuple of mocked mlflow.set_tracking_uri and mlflow.set_experiment functions.
    """
    with patch('refactor.ModelTrainer.mlflow.set_tracking_uri') as mock_set_tracking_uri, \
         patch('refactor.ModelTrainer.mlflow.set_experiment') as mock_set_experiment:
        yield mock_set_tracking_uri, mock_set_experiment

@pytest.fixture
def model_trainer(synthetic_data, mock_mlflow):
    """
    Fixture to initialize the ModelTrainer with synthetic data and mocked MLflow.
    Args:
        synthetic_data: Fixture providing synthetic data.
        mock_mlflow: Fixture providing mocked MLflow functions.
    Returns:
        Tuple containing the trainer and data splits.
    """
    x_train, x_test, y_train, y_test = synthetic_data
    model = LogisticRegression()
    params = {'C': 1.0, 'solver': 'liblinear'}
    trainer = ModelTrainer(
        model=model,
        params=params,
        model_name='LogisticRegression',
        mlflow_experiment='Test_Experiment'
    )
    return trainer, x_train, x_test, y_train, y_test, mock_mlflow  # Now returns 6 items

@patch.object(LogisticRegression, 'fit')
@patch.object(LogisticRegression, 'set_params')
def test_train_success(mock_set_params, mock_fit, model_trainer):
    """
    Test that the _train method sets parameters and fits the model correctly.
    Args:
        mock_set_params: Mock for model.set_params.
        mock_fit: Mock for model.fit.
        model_trainer: Fixture providing the ModelTrainer instance.
    """
    trainer, x_train, _, y_train, _, _ = model_trainer  # Unpack all 6 values

    # Call the _train method
    trainer._train(x_train, y_train)

    # Assert that set_params was called once with the correct parameters
    mock_set_params.assert_called_once_with(**trainer.params)

    # Assert that fit was called once with the correct training data
    mock_fit.assert_called_once_with(x_train, y_train)

def test_initialization(model_trainer):
    """
    Test that ModelTrainer initializes correctly and sets up MLflow.
    Args:
        model_trainer: Fixture providing the ModelTrainer instance.
    """
    trainer, _, _, _, _, mock_mlflow = model_trainer  # Unpack all 6 values
    mock_set_tracking_uri, mock_set_experiment = mock_mlflow

    # Assert MLflow tracking URI is set correctly
    mock_set_tracking_uri.assert_called_once_with(trainer.mlflow_tracking_uri)

    # Assert MLflow experiment is set correctly
    mock_set_experiment.assert_called_once_with(trainer.mlflow_experiment)

    # Additional assertions to verify ModelTrainer attributes
    assert trainer.model_name == 'LogisticRegression'
    assert trainer.params == {'C': 1.0, 'solver': 'liblinear'}

@patch('refactor.ModelTrainer.mlflow.log_metric')
def test_log_metrics_success(mock_log_metric, model_trainer):
    """
    Test that metrics are logged correctly using MLflow.
    Args:
        mock_log_metric: Mock for mlflow.log_metric.
        model_trainer: Fixture providing the ModelTrainer instance.
    """
    trainer, _, _, _, _, _ = model_trainer  # Unpack all 6 values
    metrics = {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.75, 'f1_score': 0.77}

    # Call the _log_metrics method
    trainer._log_metrics(metrics)

    # Assert that log_metric was called correctly for each metric
    for key, value in metrics.items():
        mock_log_metric.assert_any_call(key, value)

    # Assert that log_metric was called exactly len(metrics) times
    assert mock_log_metric.call_count == len(metrics)

def test_log_metrics_without_setup():
    """
    Test logging metrics without proper MLflow setup.
    This assumes that ModelTrainer handles such scenarios gracefully.
    """
    # Initialize ModelTrainer without proper MLflow setup
    model = LogisticRegression()
    params = {'C': 1.0, 'solver': 'liblinear'}
    trainer = ModelTrainer(
        model=model,
        params=params,
        model_name='LogisticRegression',
        mlflow_experiment='Test_Experiment'
    )

    # Define metrics
    metrics = {'accuracy': 0.85, 'precision': 0.80}

    with patch('refactor.ModelTrainer.mlflow.log_metric') as mock_log_metric:
        trainer._log_metrics(metrics)
        for key, value in metrics.items():
            mock_log_metric.assert_any_call(key, value)
        assert mock_log_metric.call_count == len(metrics)
