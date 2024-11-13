# ModelTrainer.py

import os
import sys
import logging
from typing import Any, Dict

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import xgboost  # Import xgboost to access XGBModel
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(levelname)s - %(name)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


class ModelTrainer:
    """
    A class to handle the training, evaluation, and logging of machine learning models using MLflow.
    """

    def __init__(
        self,
        model: BaseEstimator,
        params: Dict[str, Any],
        model_name: str,
        mlflow_experiment: str,
        mlflow_tracking_uri: str = "",
    ):
        """
        Initializes the ModelTrainer with the given model and parameters.

        Args:
            model (BaseEstimator): The machine learning model to be trained.
            params (Dict[str, Any]): Hyperparameters for the model.
            model_name (str): A descriptive name for the model.
            mlflow_experiment (str): The name of the MLflow experiment.
            mlflow_tracking_uri (str, optional): The MLflow tracking server URI. Defaults to "http://172.29.4.89:5000".
        """
        self.model = model
        self.params = params
        self.model_name = model_name
        self.mlflow_experiment = mlflow_experiment
        self.mlflow_tracking_uri = mlflow_tracking_uri

        # Set MLflow tracking URI
        if self.mlflow_tracking_uri != "":
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {self.mlflow_tracking_uri}")

        # Set MLflow experiment
        mlflow.set_experiment(self.mlflow_experiment)
        logger.info(f"MLflow experiment set to: {self.mlflow_experiment}")

    def _train(self, x_train: Any, y_train: Any) -> None:
        """
        Trains the machine learning model on the training data.

        Args:
            x_train (Any): Training features.
            y_train (Any): Training labels.
        """
        try:
            logger.info(f"Training {self.model_name} with parameters: {self.params}")
            self.model.set_params(**self.params)  # Ensure parameters are set
            self.model.fit(x_train, y_train)
            logger.info(f"Training completed for {self.model_name}.")
        except Exception as e:
            logger.error(f"Error during training the model: {e}")
            raise e

    def predict(self, x_test: Any) -> Any:
        """
        Generates predictions using the trained model.

        Args:
            x_test (Any): Testing features.

        Returns:
            Any: Predicted labels.
        """
        try:
            predictions = self.model.predict(x_test)
            logger.info(f"Predictions generated for {self.model_name}.")
            return predictions
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e

    def _evaluate(self, y_test: Any, y_pred: Any) -> Dict[str, float]:
        """
        Evaluates the model's performance using various metrics.

        Args:
            y_test (Any): True labels.
            y_pred (Any): Predicted labels.

        Returns:
            Dict[str, float]: A dictionary containing evaluation metrics.
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
            }
            logger.info(f"Evaluation metrics for {self.model_name}: {metrics}")
            return metrics
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise e

    def _log_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Logs the evaluation metrics to MLflow.

        Args:
            metrics (Dict[str, float]): Evaluation metrics.
        """
        try:
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
                logger.info(f"Logged {key}: {value} to MLflow.")
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")
            raise e

    def _log_artifacts(self, y_test: Any, y_pred: Any) -> None:
        """
        Logs classification reports and confusion matrices as artifacts to MLflow.

        Args:
            y_test (Any): True labels.
            y_pred (Any): Predicted labels.
        """
        try:
            # Classification Report
            report = classification_report(y_test, y_pred, output_dict=False)
            report_path = "classification_report.txt"
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            logger.info(f"Logged classification report to MLflow at {report_path}.")

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.title('Confusion Matrix')
            cm_path = "confusion_matrix.png"
            plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            plt.close()
            logger.info(f"Logged confusion matrix to MLflow at {cm_path}.")

            # Optionally, clean up local files after logging
            os.remove(report_path)
            os.remove(cm_path)
            logger.info("Cleaned up local artifact files.")
        except Exception as e:
            logger.error(f"Error logging artifacts to MLflow: {e}")
            raise e

    def _log_model(self) -> None:
        """
        Logs the trained model to MLflow.
        """
        try:
            if isinstance(self.model, xgboost.XGBModel):
                mlflow.xgboost.log_model(self.model, self.model_name)
                logger.info(f"Logged XGBoost model to MLflow under {self.model_name}.")
            else:
                mlflow.sklearn.log_model(self.model, self.model_name)
                logger.info(f"Logged Scikit-learn model to MLflow under {self.model_name}.")
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
            raise e

    def run_training(
        self, x_train: Any, x_test: Any, y_train: Any, y_test: Any
    ) -> Dict[str, float]:
        """
        Executes the training, evaluation, and logging of the model.

        Args:
            x_train (Any): Training features.
            x_test (Any): Testing features.
            y_train (Any): Training labels.
            y_test (Any): Testing labels.

        Returns:
            Dict[str, float]: Evaluation metrics.
        """
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=self.model_name) as run:
                logger.info(f"Started MLflow run: {run.info.run_id} for {self.model_name}.")

                # Log model parameters
                mlflow.log_params(self.params)
                logger.info(f"Logged parameters for {self.model_name}: {self.params}.")

                # Train the model
                self._train(x_train, y_train)

                # Make predictions
                y_pred = self.predict(x_test)

                # Evaluate the model
                metrics = self._evaluate(y_test, y_pred)

                # Log metrics
                self._log_metrics(metrics)

                # Log artifacts
                self._log_artifacts(y_test, y_pred)

                # Log the model
                self._log_model()

                logger.info(
                    f"Completed MLflow run: {run.info.run_id} for {self.model_name}."
                )

                # Return the metrics
                return metrics

        except Exception as e:
            logger.error(f"Error during the training pipeline: {e}")
            raise e
