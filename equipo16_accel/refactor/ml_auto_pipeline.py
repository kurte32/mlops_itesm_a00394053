# ml_auto_pipeline.py

import logging
import sys
from typing import Any, Dict, List

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import DataProcessor as dp	
import ModelTrainer as mt
import Visualizer as vz


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline function that processes raw data, visualizes it,
    prepares it for training, and trains various machine learning models.
    """
    try:
        logger.info("Starting the ML Auto Pipeline.")

        # Process raw data
        logger.info("Processing raw data.")
        processed_data_df = process_raw_data()

        # Visualize data
        logger.info("Visualizing data.")
        visualize_data(processed_data_df)

        # Prepare data for training, x and y
        logger.info("Preparing data for training.")
        x, y = prepare_data(processed_data_df)

        # Data splitting into training and testing sets
        logger.info("Splitting data into training and testing sets.")
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # Define models
        logger.info("Defining models and hyperparameter grids.")
        models = define_models()

        # Train and evaluate models with the training and testing sets
        logger.info("Training and evaluating models.")
        train_and_evaluate_models(models, x_train, x_test, y_train, y_test)

        logger.info("ML Auto Pipeline completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred in the ML Auto Pipeline: {e}")
        sys.exit(1)


def process_raw_data() -> dp.DataProcessor:
    """
    Processes the raw data by loading, cleaning, and performing feature engineering.
    Saves the processed data and returns it as a dataframe.

    Returns:
        dp.DataProcessor: An instance of DataProcessor with processed data.
    """
    # Create a new DataProcessor instance
    data_processor = dp.DataProcessor()

    # Load data
    data_processor.load_data()

    # Clean data
    data_processor.clean_data()

    # Feature engineering
    data_processor.feature_engineering()

    # Save processed data
    data_processor.save_processed_data()

    # Return processed data
    return data_processor


def visualize_data(data_processor: dp.DataProcessor) -> None:
    """
    Visualizes the data using histograms, KDE plots, boxplots, correlation matrix, and scatter plots.

    Args:
        data_processor (dp.DataProcessor): An instance of DataProcessor with processed data.
    """
    dataframe = data_processor.processed_data
    visualizer = vz.Visualizer()

    # Plot histograms
    visualizer.plot_histograms(dataframe, ['x', 'y', 'z'])

    # Plot KDE
    visualizer.plot_kde(dataframe, ['x', 'y', 'z'])

    # Plot boxplots
    visualizer.plot_boxplots(dataframe, 'configuración', ['x', 'y', 'z'])

    # Plot correlation matrix (only numeric columns)
    visualizer.display_summary_statistics(dataframe, ['configuración', 'pctid'], ['x', 'y', 'z'])

    # Plot vibration magnitude vs RPM
    visualizer.plot_vibration_vs_rpm(dataframe, 'pctid', 'vibration_magnitude', 'configuración')


def prepare_data(data_processor: dp.DataProcessor) -> (Any, Any):
    """
    Prepares the features and target variable for model training.

    Args:
        data_processor (dp.DataProcessor): An instance of DataProcessor with processed data.

    Returns:
        tuple: Features (X) and target (y) for training.
    """
    dataframe = data_processor.processed_data
    x = dataframe[['x', 'y', 'z', 'pctid', 'vibration_magnitude']]
    y = dataframe['wconfid']
    return x, y


def define_models() -> List[Dict[str, Any]]:
    """
    Defines the models and their hyperparameter grids for GridSearchCV.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing model configurations.
    """
    return [
        {
            'model': RandomForestClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [None, 10],
                'max_features': ['sqrt', 'log2', None]
            },
            'name': 'RandomForest_Model',
        },
        {
            'model': SVC(random_state=42),
            'param_grid': {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            },
            'name': 'SVM_Model',
        },
        {
            'model': xgb.XGBClassifier(
                use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            'param_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 6],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'name': 'XGBoost_Model',
        }
    ]


def get_model(model_name: str, params: Dict[str, Any]) -> Any:
    """
    Returns the model initialized with the given parameters based on the model name.

    Args:
        model_name (str): The name of the model.
        params (Dict[str, Any]): Hyperparameters for the model.

    Returns:
        Any: An instance of the configured model.
    """
    if model_name == 'RandomForest_Model':
        return RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features=params['max_features'],
            random_state=42
        )
    elif model_name == 'SVM_Model':
        return SVC(
            kernel=params['kernel'],
            C=params['C'],
            gamma=params['gamma'],
            random_state=42
        )
    elif model_name == 'XGBoost_Model':
        return xgb.XGBClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            learning_rate=params['learning_rate'],
            subsample=params['subsample'],
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def train_and_evaluate_models(models: List[Dict[str, Any]], x_train: Any, x_test: Any, y_train: Any, y_test: Any) -> None:
    """
    Trains and evaluates each model manually over all parameter combinations
    using ParameterGrid. Logs each model and its performance using MLflow.

    Args:
        models (List[Dict[str, Any]]): A list of model configurations.
        x_train (Any): Training features.
        x_test (Any): Testing features.
        y_train (Any): Training labels.
        y_test (Any): Testing labels.
    """
    mlflow_experiment = "Vibration_Configuration_Classification_refactorizado"

    for model_info in models:
        logger.info(f"Training model: {model_info['name']}")
        logger.info(f"Hyperparameter grid: {model_info['param_grid']}")

        for params in ParameterGrid(model_info['param_grid']):
            # Prepare data, scale if necessary (SVM model)
            if model_info['name'] == 'SVM_Model':
                scaler = StandardScaler()
                x_train_prepared = scaler.fit_transform(x_train)
                x_test_prepared = scaler.transform(x_test)
                logger.info(f"Scaled data for {model_info['name']}")
            else:
                x_train_prepared = x_train
                x_test_prepared = x_test

            # Initialize the model with current hyperparameters
            model = get_model(model_info['name'], params)

            # Train and evaluate the model using ModelTrainer
            trainer = mt.ModelTrainer(model, params, model_info['name'], mlflow_experiment)
            trainer.run_training(x_train_prepared, x_test_prepared, y_train, y_test)

            logger.info(f"Trained {model_info['name']} with params: {params}")


if __name__ == '__main__':
    main()
