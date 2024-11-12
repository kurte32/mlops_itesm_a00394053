# ml_auto_pipeline.py

import logging
import sys
from typing import Any, Dict, List, Tuple

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import random  # Importación añadida

import DataProcessor as dp
import ModelTrainer as mt
import Visualizer as vz
import joblib 
# Configure logging
from logging_config import setup_logging
logger = setup_logging()

# Establecer la semilla global para garantizar reproducibilidad
SEED = 42  

# Establecer semillas en los generadores de números aleatorios
random.seed(SEED)
np.random.seed(SEED)



def main() -> None:
    """Main pipeline function that orchestrates the entire ML process.

    This function processes raw data, visualizes it, prepares it for training,
    and trains various machine learning models.

    Raises:
        SystemExit: If an error occurs during pipeline execution.
    """
    try:
        logger.info("Starting the ML Auto Pipeline.")

        # Process raw data
        logger.info("Processing raw data.")
        data_processor = process_raw_data()

        # Visualize data
        logger.info("Visualizing data.")
        visualize_data(data_processor)

        # Prepare data for training
        logger.info("Preparing data for training.")
        x, y = prepare_data(data_processor)

        # Split data into training and testing sets
        logger.info("Splitting data into training and testing sets.")
        # Usamos random_state=SEED para asegurar reproducibilidad
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=SEED
        )

        # Define models and hyperparameter grids
        logger.info("Defining models and hyperparameter grids.")
        models = define_models()

        # Train and evaluate models
        logger.info("Training and evaluating models.")
        train_and_evaluate_models(models, x_train, x_test, y_train, y_test)

        logger.info("ML Auto Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the ML Auto Pipeline: {e}")
        sys.exit(1)

def process_raw_data() -> dp.DataProcessor:
    """Processes raw data by loading, cleaning, and performing feature engineering.

    Returns:
        dp.DataProcessor: An instance containing the processed data.

    Raises:
        ValueError: If data processing fails.
    """
    try:
        # Initialize DataProcessor
        data_processor = dp.DataProcessor()

        # Load data
        data_processor.load_data()

        # Clean data
        data_processor.clean_data()

        # Perform feature engineering
        data_processor.feature_engineering()

        # Save processed data
        data_processor.save_processed_data()

        return data_processor

    except Exception as e:
        logger.error(f"Error processing raw data: {e}")
        raise ValueError(f"Error in process_raw_data: {e}")

def visualize_data(data_processor: dp.DataProcessor) -> None:
    """Generates visualizations for exploratory data analysis.

    Args:
        data_processor (dp.DataProcessor): DataProcessor instance with processed data.

    Raises:
        ValueError: If visualization fails.
    """
    try:
        dataframe = data_processor.processed_data
        visualizer = vz.Visualizer()

        # Generate histograms
        visualizer.plot_histograms(dataframe, ['x', 'y', 'z'])

        # Generate KDE plots
        visualizer.plot_kde(dataframe, ['x', 'y', 'z'])

        # Generate boxplots
        visualizer.plot_boxplots(dataframe, 'configuración', ['x', 'y', 'z'])

        # Display summary statistics
        visualizer.display_summary_statistics(
            dataframe, ['configuración', 'pctid'], ['x', 'y', 'z']
        )

        # Plot vibration magnitude vs RPM
        visualizer.plot_vibration_vs_rpm(
            dataframe, 'pctid', 'vibration_magnitude', 'configuración'
        )

    except Exception as e:
        logger.error(f"Error during data visualization: {e}")
        raise ValueError(f"Error in visualize_data: {e}")

def prepare_data(data_processor: dp.DataProcessor) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepares features and target variable for model training.

    Args:
        data_processor (dp.DataProcessor): DataProcessor instance with processed data.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).

    Raises:
        ValueError: If data preparation fails.
    """
    try:
        if not isinstance(data_processor, dp.DataProcessor):
            raise TypeError("data_processor must be an instance of DataProcessor.")

        dataframe = data_processor.processed_data

        if dataframe is None or dataframe.empty:
            raise ValueError("Processed data is empty or None.")

        required_columns = ['x', 'y', 'z', 'pctid', 'vibration_magnitude', 'wconfid']
        if not all(column in dataframe.columns for column in required_columns):
            missing = set(required_columns) - set(dataframe.columns)
            raise ValueError(f"Dataframe is missing required columns: {missing}")

        x = dataframe[['x', 'y', 'z', 'pctid', 'vibration_magnitude']]
        y = dataframe['wconfid']
        return x, y

    except Exception as e:
        logger.error(f"Error preparing data for training: {e}")
        raise ValueError(f"Error in prepare_data: {e}")

def define_models() -> List[Dict[str, Any]]:
    """Defines machine learning models and their hyperparameter grids.

    Returns:
        List[Dict[str, Any]]: List of model configurations.

    Raises:
        ValueError: If model definitions fail.
    """
    try:
        # Establecemos random_state=SEED en los modelos para garantizar reproducibilidad
        models = [
            {
                'model': RandomForestClassifier(random_state=SEED),
                'param_grid': {
                    'n_estimators': [50, 100],
                    'max_depth': [None, 10],
                    'max_features': ['sqrt', 'log2', None]
                },
                'name': 'RandomForest_Model',
            },
            {
                'model': SVC(random_state=SEED),
                'param_grid': {
                    'kernel': ['linear', 'rbf'],
                    'C': [0.1, 1, 10],
                    'gamma': ['scale', 'auto']
                },
                'name': 'SVM_Model',
            },
            {
                'model': xgb.XGBClassifier(
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    random_state=SEED
                ),
                'param_grid': {
                    'n_estimators': [50, 100],
                    'max_depth': [3, 6],
                    'learning_rate': [0.01, 0.1],
                    'subsample': [0.8, 1.0]
                },
                'name': 'XGBoost_Model',
            }
        ]
        return models

    except Exception as e:
        logger.error(f"Error defining models: {e}")
        raise ValueError(f"Error in define_models: {e}")
    

def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    """Initializes and returns a model based on the given name and parameters.

    Args:
        model_name (str): Name of the model.
        params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
        Any: Initialized model instance.

    Raises:
        ValueError: If the model name is unknown.
    """
    try:
        if model_name == 'RandomForest_Model':
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                max_features=params['max_features'],
                random_state=SEED
            )
        elif model_name == 'SVM_Model':
            return SVC(
                kernel=params['kernel'],
                C=params['C'],
                gamma=params['gamma'],
                probability=True,  # Enable probability estimates
                random_state=SEED
            )
        elif model_name == 'XGBoost_Model':
            return xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=SEED
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    except Exception as e:
        logger.error(f"Error initializing model {model_name}: {e}")
        raise

#
def train_and_evaluate_models(
    models: List[Dict[str, Any]],
    x_train: Any,
    x_test: Any,
    y_train: Any,
    y_test: Any
) -> None:
    """Trains and evaluates models over hyperparameter grids.

    Args:
        models (List[Dict[str, Any]]): List of model configurations.
        x_train (Any): Training features.
        x_test (Any): Testing features.
        y_train (Any): Training labels.
        y_test (Any): Testing labels.

    Raises:
        ValueError: If training or evaluation fails.
    """
    try:
        mlflow_experiment = "Vibration_Configuration_Classification_refactorizado"
        best_score = -np.inf
        best_model = None
        best_model_name = ""
        best_params = {}

        for model_info in models:
            logger.info(f"Training model: {model_info['name']}")
            logger.info(f"Hyperparameter grid: {model_info['param_grid']}")

            # Iterate over all hyperparameter combinations
            for params in ParameterGrid(model_info['param_grid']):
                logger.info(f"Training with parameters: {params}")

                # Check if scaling is needed
                if model_info['name'] == 'SVM_Model':
                    # Scale features for SVM
                    scaler = StandardScaler()
                    x_train_prepared = scaler.fit_transform(x_train)
                    x_test_prepared = scaler.transform(x_test)
                    logger.debug(f"Data scaled for {model_info['name']}")
                else:
                    # Use original features
                    x_train_prepared = x_train
                    x_test_prepared = x_test

                # Initialize model
                model = get_model(model_info['name'], params)

                # Create ModelTrainer and run training
                trainer = mt.ModelTrainer(
                    model, params, model_info['name'], mlflow_experiment
                )
                metrics = trainer.run_training(x_train_prepared, x_test_prepared, y_train, y_test)

                # Assume `metrics` is a dictionary containing evaluation metrics like accuracy
                current_score = metrics.get('accuracy', 0)  # Replace 'accuracy' with your metric

                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_model_name = model_info['name']
                    best_params = params
                    print(f"Best model updated: {best_model_name} with accuracy: {best_score}")

                logger.info(f"Completed training for {model_info['name']} with params: {params}")

        if best_model is not None:
            # Save the best model
            model_filename = 'best_model.joblib'
            joblib.dump(best_model, model_filename)
            logger.info(f"Best model saved as {model_filename} with accuracy: {best_score}")
        else:
            logger.error("No model was trained successfully.")
            raise ValueError("Training failed to produce any model.")

    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        raise ValueError(f"Error in train_and_evaluate_models: {e}")

if __name__ == '__main__':
    main()
